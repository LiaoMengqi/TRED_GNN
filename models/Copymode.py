import torch
import torch.nn as nn
import numpy as np
from utils import Dataloader


class Copymode(nn.Module):
    def __init__(self, hidden_dim, data: Dataloader, n_layer):
        super(Copymode, self).__init__()
        self.num_relation = data.num_relation
        self.num_entity = data.num_entity
        self.hidden_dim = hidden_dim

        # parameter
        self.entity_embeds = nn.Parameter(torch.Tensor(self.num_entity, self.hidden_dim))
        self.relation_embeds = nn.Parameter(torch.Tensor(self.num_relation * 2 + 1, self.hidden_dim))
        self.penalty = nn.Parameter(torch.Tensor(n_layer + 1, 1))
        self.W_s = nn.Linear(self.hidden_dim * 2, self.num_entity)
        self.init_parameters()

        # active function
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.entity_embeds,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.relation_embeds,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.trunc_normal_(self.penalty, mean=0.5, std=1.0, a=0.2, b=0.8)

    def forward(self, data_batched, vocabulary_sampled_list, vocabulary_all_sampled):
        n_layer = len(vocabulary_sampled_list)
        sub_embed = self.entity_embeds[data_batched[:, 0]]
        rela_embed = self.relation_embeds[data_batched[:, 1]]
        matrix = torch.cat((sub_embed, rela_embed), dim=1)
        scores = self.tanh(self.W_s(matrix))
        encoded_mask = torch.zeros_like(scores)
        for i in range(n_layer):
            encoded_mask = encoded_mask + torch.Tensor(
                np.array(vocabulary_sampled_list[i] != 0, dtype=float)).cuda() * (self.penalty[i])

        encoded_mask = encoded_mask + torch.Tensor(np.array(vocabulary_all_sampled == 0, dtype=float)).cuda() * (
                    -self.penalty[-1])
        scores = scores + encoded_mask
        return scores
