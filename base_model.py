import models
import utils
import torch
from tqdm import tqdm
import numpy as np
import sys


class BaseModel(object):
    def __init__(self, opts):
        self.n_layer = opts.n_layer
        self.data = utils.Dataloader(opts.path)
        self.batch_size = opts.batch_size
        self.data.load_tkg()
        self.tred_gnn = models.TRED_GNN(self.data, opts)
        self.tred_gnn.cuda()
        self.optimizer = torch.optim.Adam(self.tred_gnn.parameters(), lr=opts.lr, weight_decay=opts.lamb)
        self.train_history = []

    def train(self, output=True):
        if self.data.time_length_train - self.n_layer < 0:
            raise Exception('Error!')
        self.tred_gnn.train()
        loss_history = []
        # for time_stamp in tqdm(range(self.n_layer, self.n_layer + 100), file=sys.stdout):
        for time_stamp in tqdm(range(self.n_layer, self.data.time_length_train), file=sys.stdout):
            num_query = self.data.data_splited[time_stamp].shape[0]
            num_batch = num_query // self.batch_size + (num_query % self.batch_size > 0)
            for i in range(num_batch):
                indexes = range(i * self.batch_size, min((i + 1) * self.batch_size, num_query))
                data_batched = torch.tensor(self.data.get_batch(time_stamp, indexes)).cuda()

                self.tred_gnn.zero_grad()
                scores = self.tred_gnn.forward(time_stamp, data_batched[:, 0], data_batched[:, 1])

                pos_scores = scores[[torch.arange(len(scores)), data_batched[:, 2]]]
                max_n = torch.max(scores, 1, keepdim=True)[0]
                loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n), 1)))
                loss.backward()
                loss_history.append(loss.item())
                self.optimizer.step()

                # acoid NaN
                for para in self.tred_gnn.parameters():
                    para_data = para.data.clone()
                    flag = para_data != para_data
                    para_data[flag] = np.random.random()
                    para.data.copy_(para_data)

        loss_average = sum(loss_history) / len(loss_history)
        mrr, ht1, ht10 = self.evaluate()
        self.train_history.append((loss_average, mrr, ht1, ht10))

        if output:
            print('loss : ', loss_average, ' mrr : ', mrr, ' hist@1 : ', str(ht1), ' hist@10 : ' + str(ht10))

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

        self.tred_gnn.eval()
        ranks = []
        for time_stamp in tqdm(range(start_time_stamp, end_time_stamp), file=sys.stdout):
            # for time_stamp in range(start_time_stamp, start_time_stamp + 5):
            num_query = self.data.data_splited[time_stamp].shape[0]
            num_batch = num_query // self.batch_size + (num_query % self.batch_size > 0)

            for i in range(num_batch):
                indexes = range(i * self.batch_size, min((i + 1) * self.batch_size, num_query))
                data_batched = torch.tensor(self.data.get_batch(time_stamp, indexes)).cuda()
                scores = self.tred_gnn.forward(time_stamp, data_batched[:, 0], data_batched[:, 1])
                scores = scores.data.cpu().numpy()
                rank = utils.cal_ranks(scores, data_batched[:, 2].data.cpu().numpy())
                ranks = ranks + rank

        ranks = np.array(ranks)
        mrr, ht1, ht10 = utils.cal_performance(ranks)
        return mrr, ht1, ht10
