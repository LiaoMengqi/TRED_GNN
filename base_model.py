import utils
import torch
from tqdm import tqdm
import numpy as np
import sys
from utils import get_logger
import json
import time
from models import model_dict
from scipy.sparse import csr_matrix
import torch.nn as nn


class Trainer(object):
    def __init__(self, opts):
        self.opts = opts
        self.n_layer = opts.n_layer
        self.data = utils.Dataloader(opts.path)
        self.batch_size = opts.batch_size
        self.data.load_tkg()
        self.model_name = opts.model_name

        # copy mode
        self.alpha = opts.alpha
        self.copymode = model_dict["Copymode"](opts.copy_hidden_dim, self.data, self.n_layer)
        self.vocabulary = []
        self.vocabulary_all = csr_matrix(([], ([], [])), shape=(
            self.data.num_entity * (self.data.num_relation * 2 + 1), self.data.num_entity))
        self.copymode.cuda()

        # tred_gnn
        self.tred_gnn = model_dict[self.model_name](self.data, opts)
        self.tred_gnn.cuda()

        self.optimizer = torch.optim.Adam(self.tred_gnn.parameters(), lr=opts.lr, weight_decay=opts.lamb)
        self.sigmoid = nn.Sigmoid()

        self.train_history = []
        self.loss_history = []
        if opts.tag is None or len(opts.tag) == 0:
            self.result_dir = f"results/{opts.model_name}/ICEWS14s"
        else:
            self.result_dir = f"results/{opts.model_name}/{opts.tag}/ICEWS14s"
        self.logger = get_logger(self.result_dir + "/log.txt")
        self.now_epoch = 0
        self.active = nn.Tanh()
        self.logger.info(json.dumps(opts))

    def train_epoch(self):
        self.reset_vocabulary()
        self.now_epoch += 1
        self.logger.info(f"Start epoch {self.now_epoch} train")
        if self.data.time_length_train - self.n_layer < 0:
            raise Exception('Error!')
        self.tred_gnn.train()
        self.copymode.train()
        start_time = time.time()
        # for time_stamp in tqdm(range(self.n_layer, self.n_layer + 50), file=sys.stdout):
        for time_stamp in tqdm(range(self.n_layer, self.data.time_length_train), file=sys.stdout,disable=self.opts.disable_bar):
            if time_stamp > self.n_layer:
                self.update_vocabulary(time_stamp - 1)
            num_query = self.data.data_splited[time_stamp].shape[0]
            num_batch = num_query // self.batch_size + (num_query % self.batch_size > 0)
            for i in range(num_batch):
                indexes = range(i * self.batch_size, min((i + 1) * self.batch_size, num_query))
                data_batched = torch.tensor(self.data.get_batch(time_stamp, indexes)).cuda()

                self.tred_gnn.zero_grad()
                scores = self.tred_gnn.forward(time_stamp, data_batched[:, 0], data_batched[:, 1])

                self.copymode.zero_grad()
                vocabulary_sampled_list, vocabulary_all_sampled = self.vocabulary_sample(data_batched)
                scores_copy = self.copymode.forward(data_batched, vocabulary_sampled_list, vocabulary_all_sampled)

                scores_composed = self.compose_scores(scores, scores_copy)

                loss = self.cal_loss(data_batched, scores_composed)

                loss.backward()

                self.loss_history.append(loss.item())
                self.optimizer.step()

                # avoid NaN
                for para in self.tred_gnn.parameters():
                    para_data = para.data.clone()
                    flag = para_data != para_data
                    para_data[flag] = np.random.random()
                    para.data.copy_(para_data)
        evaluate_time = time.time()
        self.logger.info(f"Start epoch {self.now_epoch} evaluate")
        v_mrr, v_h1, v_h3, v_h10, v_h100 = self.evaluate()
        # t_mrr, t_h1, t_h3, t_h10, t_h100 = self.evaluate(data_eval="test")
        finish_time = time.time()
        result = {
            "loss_train": sum(self.loss_history) / len(self.loss_history),
            "v_mrr": v_mrr,
            "v_h1": v_h1,
            "v_h3": v_h3,
            "v_h10": v_h10,
            "v_h100": v_h100,
            # "t_mrr": t_mrr,
            # "t_h1": t_h1,
            # "t_h3": t_h3,
            # "t_h10": t_h10,
            # "t_h100": t_h100,
            "time_train": evaluate_time - start_time,
            "time_valid": finish_time - evaluate_time
        }
        self.train_history.append(result)
        self.logger.info(f"Finish epoch {self.now_epoch}, result:")
        self.logger.info(json.dumps(result))

    def evaluate(self, data_eval='valid'):
        if data_eval == 'valid':
            start_time_stamp = self.data.time_length_train + self.n_layer
            end_time_stamp = self.data.time_length_train + self.data.time_length_valid
        elif data_eval == 'test':
            start_time_stamp = self.data.time_length_train + self.data.time_length_valid + self.n_layer
            end_time_stamp = self.data.time_length
        else:
            raise Exception('Error!')
        if start_time_stamp >= end_time_stamp:
            raise Exception('Error!')
        self.reset_vocabulary(start_time_stamp - self.n_layer, update_all=False)
        self.tred_gnn.eval()
        self.copymode.eval()
        ranks = []
        # for time_stamp in tqdm(range(start_time_stamp, start_time_stamp + 10), file=sys.stdout):
        for time_stamp in tqdm(range(start_time_stamp, end_time_stamp), file=sys.stdout, disable=self.opts.disable_bar):
            num_query = self.data.data_splited[time_stamp].shape[0]
            num_batch = num_query // self.batch_size + (num_query % self.batch_size > 0)
            if time_stamp > start_time_stamp:
                self.update_vocabulary(time_stamp - 1, update_all=False)
            for i in range(num_batch):
                indexes = range(i * self.batch_size, min((i + 1) * self.batch_size, num_query))
                data_batched = torch.tensor(self.data.get_batch(time_stamp, indexes)).cuda()
                scores = self.tred_gnn.forward(time_stamp, data_batched[:, 0], data_batched[:, 1])

                vocabulary_sampled_list, vocabulary_all_sampled = self.vocabulary_sample(data_batched)
                scores_copy = self.copymode.forward(data_batched, vocabulary_sampled_list, vocabulary_all_sampled)

                scores_composed = self.compose_scores(scores, scores_copy)

                scores_composed = scores_composed.data.cpu().numpy()
                rank = utils.cal_ranks(scores_composed, data_batched[:, 2].data.cpu().numpy())
                ranks = ranks + rank

        ranks = np.array(ranks)
        mrr, h_1, h_3, h_10, h_100 = utils.cal_performance(ranks)
        return mrr, h_1, h_3, h_10, h_100

    def cal_loss(self, data_batched, scores):
        pos_scores = scores[[torch.arange(len(scores)), data_batched[:, 2]]]
        max_n = torch.max(scores, 1, keepdim=True)[0]
        loss = torch.sum(- pos_scores + max_n[:, 0] + torch.log(torch.sum(torch.exp(scores - max_n), 1)))
        return loss

    def update_vocabulary(self, time_stamp, remove=True, update_all=True):
        """
        更新 vocabulary,将时间 time_stamp 的查询 (s,r,?) 的目标实体 o 记录下来
        """
        row = self.data.data_splited[time_stamp][:, 0] * self.data.num_relation + self.data.data_splited[
                                                                                      time_stamp][:, 1]
        col = self.data.data_splited[time_stamp][:, 2]
        data = np.ones(len(row))
        vocabulary_t = csr_matrix((data, (row, col)), shape=(
            self.data.num_entity * (self.data.num_relation * 2 + 1), self.data.num_entity))
        if update_all:
            self.vocabulary_all = self.vocabulary_all + vocabulary_t

        self.vocabulary.append(vocabulary_t)

        if remove:
            self.vocabulary.pop(0)

    def reset_vocabulary(self, start=0, update_all=True):
        """
        重置 vocabulary
        """
        self.vocabulary = []
        for time_stamp in range(start, start + self.n_layer):
            self.update_vocabulary(time_stamp, False, update_all)

    def vocabulary_sample(self, data_batched):
        vocabulary_index = (data_batched[:, 0] * self.data.num_relation + data_batched[:, 1]).cpu()
        vocabulary_sampled_list = []
        for i in range(self.n_layer):
            vocabulary_sampled = torch.Tensor(self.vocabulary[i][vocabulary_index].todense())
            one_hot_vocabulary = vocabulary_sampled.masked_fill(vocabulary_sampled != 0, 1)
            vocabulary_sampled_list.append(one_hot_vocabulary)
        vocabulary_all_sampled = torch.Tensor(self.vocabulary_all[vocabulary_index].todense())
        vocabulary_all_sampled = vocabulary_all_sampled.masked_fill(vocabulary_all_sampled != 0, 1)
        return vocabulary_sampled_list, vocabulary_all_sampled

    def compose_scores(self, scores, scores_copy):

        # scores_composed = scores * (self.alpha * scores_copy + 1)
        # 加法
        # scores = self.sigmoid(scores)
        # scores_copy = self.sigmoid(scores_copy)
        scores_composed = scores * self.alpha + scores_copy*(1-self.alpha)
        return scores_composed

    def process_results(self):
        with open(self.result_dir + "/history.json", "w") as f:
            json.dump(self.train_history, f)
        best_result = sorted(self.train_history, key=lambda x: x["v_mrr"], reverse=True)
        self.logger.info("Finish all epoch, the best is " + str(best_result))
