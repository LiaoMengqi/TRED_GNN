import torch
from scipy.sparse import csr_matrix
import numpy as np
from scipy.stats import rankdata
import copy
import subprocess
from typing import Optional,List,Dict

import logging
import os
import re

def get_logger(log_filename: str):
    """指定保存日志的文件路径，日志级别，以及调用文件
        将日志存入到指定的文件中
        :paramlogger:
        """
    # 创建一个logger
    logger = logging.getLogger("Train logger")
    logger.setLevel(logging.INFO)
    # 此处的判断是为了不重复调用test_log，导致重复打印出日志；第一次调用就会创建一个，第二次就不会再次调用了，也就不会出现重复日志的情况

    # 创建一个handler，用于写入日志文件
    if len(log_filename) == 0 or log_filename[-1] in ('/', '\\'):
        raise FileNotFoundError("无效的log文件地址,请使用文件而不是目录当作log输出地")
    father_dir = os.path.dirname(os.path.abspath(log_filename))
    if not os.path.exists(father_dir):
        os.makedirs(father_dir, exist_ok=True)
    fh = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    # 创建一个hander用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 定义handler的输出格式
    formeter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s')
    fh.setFormatter(formeter)
    ch.setFormatter(formeter)
    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

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

    def _read_fact(self, file_name:str):
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
        """
        因为时间的连续性问题，按照传统的思路应该是根据前几个时间片来考虑未来，所以有些类似于inductive实验
        问题1：
        不过，这种模式并没有考虑到长期的历史情况。
        比如，
        正面案例1：
        美国打击了巴基斯坦的外汇市场（假如），近期巴基斯坦财政危机，急需国外政府支援。联系了中国，此时巴基斯坦下一步会寻求谁的支援？
        答案是：中国
        反面案例1：
        美国打击了巴基斯坦的外汇市场（假如），近期巴基斯坦财政危机，急需国外政府支援，此时巴基斯坦下一步会寻求谁的支援？
        答案是：美国
        分析：因为近期的国家实体只有美国，所以只能推导出美国。但是如果考虑长期的历史信息（长历史依赖，历史表示）或是静态关系知识，则可以推导出是中国。正确的链条是巴基斯坦和中国友好，所以遇到困难时会找中国。
        问题2：
        历史记忆保留问题，
        推理时只使用了最后一层的实体，而没有考虑更近的实体。
        """
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


class ElasticEmbedding:
    """"
    用于组织稀疏张量的数据结构，用于长期记忆出现过的实体的表示。
    """

    def __init__(self, batch_size: int, entity_num: int, hidden_size: int, device):
        self.index_matrix = - np.ones(shape=(batch_size, entity_num), dtype=np.int64)
        self.hidden_size = hidden_size
        # 关于数据是否需要设置可反向传播
        self.data = torch.zeros(batch_size, hidden_size).to(device)
        self.device = device

    def get(self, index:np.ndarray)->torch.Tensor:
        """
        按照索引获取节点的隐藏表示
        :param index: array(list) of [sample_id, entity_id]
        :return: Tensor, size=(len(index), hidden_size),获取的数据
        """
        _index = self.index_matrix[index[:, 0], index[:, 1]]
        effective_mask = _index > -1
        effective_index = _index[effective_mask]
        temp = torch.zeros(len(index), self.hidden_size).to(self.device)
        temp[effective_mask] = self.data[effective_index]
        return temp

    def set(self, index:np.ndarray, data: torch.Tensor)->None:
        """
        将节点的表示更新到包含所有的节点表示的表中
        :param index:
        :param data:
        :return:
        """
        _index = self.index_matrix[index[:, 0], index[:, 1]]
        effective_mask = _index > -1
        missing_mask = effective_mask == False

        # 先创建缺失的索引
        missing_index = index[missing_mask]
        self.index_matrix[missing_index[:, 0], missing_index[:, 1]] = \
            np.arange(len(self.data), len(self.data) + len(missing_index))

        effective_index = _index[effective_mask]
        # 能找到的直接赋值操作
        self.data[effective_index] = data[effective_mask]
        # 找不到的
        missing_data = data[missing_mask]
        self.data = torch.vstack((self.data, missing_data))
        
    def get_all(self):
        valid_mask = self.index_matrix>-1
        indices = self.index_matrix[valid_mask]
        data = self.data[indices]
        return *valid_mask.nonzero(),data


def gpu_setting():
    try:
        gpu = select_gpu()
    except UnicodeDecodeError:
        gpu = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        print('gpu:', gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return device,gpu


def cal_ranks(scores, labels):
    row_index = np.arange(scores.shape[0])
    scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    rank = rankdata(-scores, axis=1)
    rank = rank[row_index, labels]
    return list(rank)


def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks <= 1) * 1.0 / len(ranks)
    h_3 = sum(ranks <= 3) * 1.0 / len(ranks)
    h_10 = sum(ranks <= 10) * 1.0 / len(ranks)
    h_100 = sum(ranks <= 100) * 1.0 / len(ranks)
    return mrr, h_1, h_3, h_10, h_100

def select_gpu():
    try:
        nvidia_info = subprocess.run(
            'nvidia-smi', stdout=subprocess.PIPE).stdout.decode()
    except UnicodeDecodeError:
        nvidia_info = subprocess.run(
            'nvidia-smi', stdout=subprocess.PIPE).stdout.decode("gbk")
    used_list = re.compile(r"(\d+)MiB\s+/\s+\d+MiB").findall(nvidia_info)
    used = [(idx, int(num)) for idx, num in enumerate(used_list)]
    sorted_used = sorted(used, key=lambda x: x[1])
    return sorted_used[0][0]

class EnhancedDict(dict):
    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(__self, '__key', kwargs.pop('__key', None))
        object.__setattr__(__self, '__frozen', False)
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in kwargs.items():
            __self[key] = __self._hook(val)

    def __setattr__(self, name, value):
        if hasattr(self.__class__, name):
            raise AttributeError("'Dict' object attribute "
                                 "'{0}' is read-only".format(name))
        else:
            self[name] = value

    def __setitem__(self, name, value):
        isFrozen = (hasattr(self, '__frozen') and
                    object.__getattribute__(self, '__frozen'))
        if isFrozen and name not in super(EnhancedDict, self).keys():
            raise KeyError(name)
        super(EnhancedDict, self).__setitem__(name, value)
        try:
            p = object.__getattribute__(self, '__parent')
            key = object.__getattribute__(self, '__key')
        except AttributeError:
            p = None
            key = None
        if p is not None:
            p[key] = self
            object.__delattr__(self, '__parent')
            object.__delattr__(self, '__key')

    def __add__(self, other):
        if not self.keys():
            return other
        else:
            self_type = type(self).__name__
            other_type = type(other).__name__
            msg = "unsupported operand type(s) for +: '{}' and '{}'"
            raise TypeError(msg.format(self_type, other_type))

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __missing__(self, name):
        if object.__getattribute__(self, '__frozen'):
            raise KeyError(name)
        return self.__class__(__parent=self, __key=name)

    def __delattr__(self, name):
        del self[name]

    def to_dict(self):
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    item.to_dict() if isinstance(item, type(self)) else
                    item for item in value)
            else:
                base[key] = value
        return base

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def update(self, *args, **kwargs):
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError()
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if ((k not in self) or
                    (not isinstance(self[k], dict)) or
                    (not isinstance(v, dict))):
                self[k] = v
            else:
                self[k].update(v)

    def __getnewargs__(self):
        return tuple(self.items())

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def __or__(self, other):
        if not isinstance(other, (EnhancedDict, dict)):
            return NotImplemented
        new = EnhancedDict(self)
        new.update(other)
        return new

    def __ror__(self, other):
        if not isinstance(other, (EnhancedDict, dict)):
            return NotImplemented
        new = EnhancedDict(other)
        new.update(self)
        return new

    def __ior__(self, other):
        self.update(other)
        return self

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    def freeze(self, shouldFreeze=True):
        object.__setattr__(self, '__frozen', shouldFreeze)
        for key, val in self.items():
            if isinstance(val, EnhancedDict):
                val.freeze(shouldFreeze)

    def unfreeze(self):
        self.freeze(False)