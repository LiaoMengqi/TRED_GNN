import torch
from scipy.sparse import csr_matrix
import numpy as np
from scipy.stats import rankdata


class Dataloader(object):
    def __init__(self, path):
        self.path = path

    def load_tkg(self):
        self._load()
        self._split_data()
        self._load_graph()

    def _read_dict(self, file_name):
        id_dict = {}
        id_list = []
        with open(self.path + file_name, encoding='utf-8') as file:
            content = file.read()
            content = content.strip()
            content = content.split("\n")
            for line in content:
                line = line.strip()
                line = line.split('\t')
                id_dict[line[0]] = int(line[1])
                id_list.append(line[0])
        return id_dict, id_list

    def _read_fact(self, file_name):
        facts = []
        with open(self.path + file_name, encoding='utf-8') as f:
            content = f.read()
            content = content.strip()
            content = content.split("\n")
            for line in content:
                fact = line.split()
                facts.append([int(fact[0]), int(fact[1]), int(fact[2]), fact[3]])
                # reverse
                facts.append([int(fact[2]), int(fact[1]) + self.num_relation, int(fact[0]), fact[3]])
        return facts

    def _load(self):
        self.entity2id, self.id2entity = self._read_dict('entity2id.txt')
        self.num_entity = len(self.id2entity)

        self.relation2id, self.id2relation = self._read_dict('relation2id.txt')
        self.num_relation = len(self.id2relation)

        # reverse
        reverse_rela = []
        for rela in self.id2relation:
            reverse_rela.append(rela + '_reverse')
        for rela_reverse in reverse_rela:
            self.id2relation.append(rela_reverse)

        # self loop
        self.id2relation.append('idd')
        self.num_relation_extended = len(self.id2relation)

        self.data_train = self._read_fact('train.txt')
        self.data_valid = self._read_fact('valid.txt')
        self.data_test = self._read_fact('test.txt')

    def _split_by_time(self, data):
        time_list = []
        for fact in data:
            if fact[3] not in self.time_dict.keys():
                self.time_dict[fact[3]] = len(self.time_dict)
                time_list.append(self.time_dict[fact[3]])
                self.data_splited.append([])
            else:
                self.data_splited[self.time_dict[fact[3]]].append([fact[0], fact[1], fact[2]])
        return time_list

    def _split_data(self):
        self.time_dict = {}
        self.data_splited = []

        self.time_list_train = self._split_by_time(self.data_train)
        self.time_length_train = len(self.time_list_train)

        self.time_list_valid = self._split_by_time(self.data_valid)
        self.time_length_valid = len(self.time_list_valid)

        self.time_list_test = self._split_by_time(self.data_test)
        self.time_length_test = len(self.time_list_test)

        self.time_length = self.time_length_train + self.time_length_valid + self.time_length_test
        # list to narray
        for i in range(self.time_length):
            self.data_splited[i] = np.array(self.data_splited[i], dtype='int64')

    def _load_graph(self):
        self.fact_sub_matrix = []
        self.graph_extended = []
        for i in range(self.time_length):
            idd = np.concatenate(
                [np.expand_dims(np.arange(self.num_entity), 1),
                 2 * self.num_relation * np.ones((self.num_entity, 1), dtype='int64'),
                 np.expand_dims(np.arange(self.num_entity), 1)], 1)
            KG = np.concatenate([self.data_splited[i], idd], 0)
            self.graph_extended.append(KG)
            num_fact = KG.shape[0]
            fsm = csr_matrix((np.ones(num_fact, ), (np.arange(num_fact), KG[:, 0])),
                             shape=(num_fact, self.num_entity))
            self.fact_sub_matrix.append(fsm)

    def get_batch(self, time_stamp, index):
        return self.data_splited[time_stamp][index]

    def get_neighbors(self, nodes, time_stamp):
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])),
                               shape=(self.num_entity, nodes.shape[0]))
        edge_1hot = self.fact_sub_matrix[time_stamp].dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), self.graph_extended[time_stamp][edges[0]]],
                                       axis=1)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
        mask = sampled_edges[:, 2] == (self.num_relation * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        return tail_nodes, sampled_edges, old_nodes_new_idx


def cal_ranks(scores, labels):
    row_index = np.arange(scores.shape[0])
    scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    rank = rankdata(-scores, axis=1)
    rank = rank[row_index, labels]
    return list(rank)


def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks <= 1.0) / len(ranks)
    h_10 = sum(ranks <= 10.0) / len(ranks)
    return mrr, h_1, h_10
