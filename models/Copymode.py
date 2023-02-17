import torch
import torch.nn as nn
import numpy as np
from utils import Dataloader


class Copymode(nn.Module):
    def __init__(self, hidden_dim, data: Dataloader):
        super(Copymode, self).__init__()
        self.num_relation = data.num_relation
        self.num_entity = data.num_entity
        self.hidden_dim = hidden_dim

        self.entity_embeds = nn.Parameter(torch.Tensor(self.num_entity, self.hidden_dim))
        self.relation_embeds = nn.Parameter(torch.Tensor(self.num_relation * 2 + 1, self.hidden_dim))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.W_s = nn.Linear(self.hidden_dim * 2, self.num_entity)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.entity_embeds,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.relation_embeds,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, data_batched, time_stamp, vocabulary):
        sub_idx = data_batched[:, 0]
        rela_idx = data_batched[:, 1]
        sub_embed = self.entity_embeds[sub_idx]
        rela_embed = self.relation_embeds[rela_idx]
        m_t = torch.cat((sub_embed, rela_embed), dim=1)
        q_s = self.tanh(self.W_s(m_t))
        encoded_mask = torch.Tensor(np.array(vocabulary.cpu() == 0, dtype=float) * (-100)).cuda()
        scores = q_s + encoded_mask

        return scores
