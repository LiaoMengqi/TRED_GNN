import torch
import torch.nn as nn
from torch_scatter import scatter
from utils import Dataloader, ElasticEmbedding, EnhancedDict


class TRED_GNN(nn.Module):
    def __init__(self, data:Dataloader, params:EnhancedDict):
        super(TRED_GNN, self).__init__()
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
        """
        如何处理断掉的历史记忆？
        query head -> entity 1 -> entity 2 ->entity 3
        此时只能推理出3，不能推理1和2
        query head -> entity 1 -> entity 2 ->entity 1
        此时能推理出1，不能推理2.同时新的1保留了旧的1的信息，大概不需要考虑这种情况
         """
        num_query = subject.shape[0]
        nodes = torch.cat([torch.arange(num_query).unsqueeze(1).cuda(), subject.unsqueeze(1)], 1)
        hidden = torch.zeros(num_query, self.hidden_dim).cuda()
        h0 = torch.zeros((1, num_query, self.hidden_dim)).cuda()
        data_manager = ElasticEmbedding(num_query,self.data.num_entity,self.hidden_dim,device=h0.device)
        for i in range(self.n_layer):
            nodes, edges, idx = self.data.get_neighbors(nodes.data.cpu().numpy(), time_stamp - self.n_layer + i)
            hidden = self.gnn_layers[i](subject, relation, hidden, edges, nodes.size(0), idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            # 将获得的实体的表示存储起来
            # 这里其实应该再加一个历史衰减，随着历史的演进，离预测时间点越远的事件影响力越小。
            # 此外，这里也不能直接替换，因为旧的实体的表示也代表了一种可能的交互关系，不能直接舍弃
            # 例如：query head -r_1-> entity 1
            # query head -> entity 3 -> entity 2 -r_3->entity 1
            # 此外，还有一个问题，就是旧有的方式只让最近的时间片中的节点搜索邻居，而历史交互过的不再搜索邻居，这也是个问题。
            # 方案：给旧的实体添加真正的自环关系
            data_manager.set(nodes.cpu().numpy(), hidden)

        index_raw,index_col,hidden = data_manager.get_all()
        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros(size=(num_query, self.num_entity)).cuda()
        scores_all[[index_raw, index_col]] = scores
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
