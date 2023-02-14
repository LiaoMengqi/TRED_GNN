import models
import utils
import torch
from tqdm import tqdm
import numpy as np
import sys
from utils import get_logger
import json
import time
from models import model_dict

class Trainer(object):
    def __init__(self, opts):
        self.opts = opts
        self.n_layer = opts.n_layer
        self.data = utils.Dataloader(opts.path)
        self.batch_size = opts.batch_size
        self.data.load_tkg()
        self.model_name = opts.model_name
        
        self.tred_gnn = model_dict[self.model_name](self.data, opts)
        self.tred_gnn.cuda()
        self.optimizer = torch.optim.Adam(self.tred_gnn.parameters(), lr=opts.lr, weight_decay=opts.lamb)
        self.train_history = []
        self.loss_history = []
        if opts.tag is None or len(opts.tag)==0:
            self.result_dir = f"results/{opts.model_name}/ICEWS14s"
        else:
            self.result_dir = f"results/{opts.model_name}/{opts.tag}/ICEWS14s"
        self.logger = get_logger(self.result_dir+"/log.txt")
        self.now_epoch=0
        
        self.logger.info(json.dumps(opts))

    def train_epoch(self):
        self.now_epoch+=1
        self.logger.info(f"Start epoch {self.now_epoch} train")
        if self.data.time_length_train - self.n_layer < 0:
            raise Exception('Error!')
        self.tred_gnn.train()
        start_time = time.time()
        # for time_stamp in tqdm(range(self.n_layer, self.n_layer + 40), file=sys.stdout):
        for time_stamp in tqdm(range(self.n_layer, self.data.time_length_train), file=sys.stdout, disable=self.opts.disable_bar):
            num_query = self.data.data_splited[time_stamp].shape[0]
            num_batch = num_query // self.batch_size + (num_query % self.batch_size > 0)
            for i in range(num_batch):
                indexes = range(i * self.batch_size, min((i + 1) * self.batch_size, num_query))
                data_batched = torch.tensor(self.data.get_batch(time_stamp, indexes)).cuda()

                self.tred_gnn.zero_grad()
                scores = self.tred_gnn.forward(time_stamp, data_batched[:, 0], data_batched[:, 1])

                loss = self.cal_loss(data_batched, scores)
                loss.backward()
                self.loss_history.append(loss.item())
                self.optimizer.step()

                # acoid NaN
                for para in self.tred_gnn.parameters():
                    para_data = para.data.clone()
                    flag = para_data != para_data
                    para_data[flag] = np.random.random()
                    para.data.copy_(para_data)
        evaluate_time = time.time()
        self.logger.info(f"Start epoch {self.now_epoch} evaluate")
        v_mrr, v_h1, v_h3, v_h10, v_h100 = self.evaluate()
        t_mrr, t_h1, t_h3, t_h10, t_h100 = self.evaluate(data_eval="test")
        finish_time = time.time()
        result = {
            "loss_train":sum(self.loss_history)/len(self.loss_history),
            "v_mrr":v_mrr,
            "v_h1":v_h1,
            "v_h3":v_h3,
            "v_h10":v_h10,
            "v_h100":v_h100,
            "t_mrr":t_mrr,
            "t_h1":t_h1,
            "t_h3":t_h3,
            "t_h10":t_h10,
            "t_h100":t_h100,
            "time_train":evaluate_time-start_time,
            "time_valid":finish_time-evaluate_time
        }
        self.train_history.append(result)
        self.logger.info(f"Finish epoch {self.now_epoch}, result:")
        self.logger.info(json.dumps(result))

    def cal_loss(self, data_batched, scores):
        pos_scores = scores[[torch.arange(len(scores)), data_batched[:, 2]]]
        max_n = torch.max(scores, 1, keepdim=True)[0]
        loss = torch.sum(- pos_scores + max_n[:,0] + torch.log(torch.sum(torch.exp(scores - max_n), 1)))
        return loss

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
        # for time_stamp in tqdm(range(start_time_stamp, start_time_stamp + 5), file=sys.stdout):
        for time_stamp in tqdm(range(start_time_stamp, end_time_stamp), file=sys.stdout, disable=self.opts.disable_bar):
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
        mrr, h_1, h_3, h_10, h_100 = utils.cal_performance(ranks)
        return mrr, h_1, h_3, h_10, h_100
    
    def process_results(self):
        with open(self.result_dir+"/history.json","w") as f:
            json.dump(self.train_history,f)
        best_result = sorted(self.train_history,key=lambda x:x["v_mrr"],reverse=True)
        self.logger.info("Finish all epoch, the best is "+str(best_result))