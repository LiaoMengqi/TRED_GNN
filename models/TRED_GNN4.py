"""
这一版对原版做了简答的修改，添加了数据管理组件，能够保留旧有的节点。
缺陷：
1. 单个时间片获得的线索太少，最终的覆盖率不够
应对策略：
1. 单个时间片搜索多次

注：还没写
"""



import torch
import torch.nn as nn
from torch_scatter import scatter
from utils import Dataloader, EnhancedDict


class TRED_GNN4(nn.Module):
    def __init__(self, data:Dataloader, params:EnhancedDict):
        super(TRED_GNN4, self).__init__()
        # 超参数
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attention_dim = params.attention_dim
        self.data = data
        self.num_relation = data.num_relation
        self.num_entity = data.num_entity
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]
        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(
                GNNLayer(self.hidden_dim, self.hidden_dim, self.attention_dim, self.num_relation, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, time_stamp, subject, relation):
        num_query = subject.shape[0]
        nodes = torch.cat([torch.arange(num_query).unsqueeze(1).cuda(), subject.unsqueeze(1)], 1)
        hidden = torch.zeros(num_query, self.hidden_dim).cuda()
        h0 = torch.zeros((1, num_query, self.hidden_dim)).cuda()

        for i in range(self.n_layer):
            nodes, edges, idx = self.data.get_neighbors(nodes.data.cpu().numpy(), time_stamp - self.n_layer + i)
            hidden = self.gnn_layers[i](subject, relation, hidden, edges, nodes.size(0), idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((num_query, self.num_entity)).cuda()
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all


class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x: x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new
