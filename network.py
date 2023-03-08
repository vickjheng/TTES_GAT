import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


class EdgeBatchNorm(nn.Module):
    def __init__(self,
                 dim,
                 device):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(dim)
        self.to(device)

    def forward(self, x):
        out = x.transpose(1, 2).contiguous()
        out = self.batch_norm(out).transpose(1, 2).contiguous()

        return out
class QNetwork_conv(nn.Module):
    def __init__(self,
                 layer_dim,
                 device):
        super().__init__()
        self.hid_layer_1 = nn.Conv2d(in_channels = layer_dim[0], out_channels = layer_dim[1], kernel_size = 1)
        self.to(device)
        
    def forward(self, state):
        out = torch.tanh(self.hid_layer_1(state))
        return out
    
class QNetwork(nn.Module):
    def __init__(self,
                 layer_dim,
                 device):
        super().__init__()
        self.hid_layer_1 = nn.Linear(layer_dim[0], layer_dim[1])
        self.hid_layer_2 = nn.Linear(layer_dim[1], layer_dim[2])
        self.hid_layer_3 = nn.Linear(layer_dim[2], layer_dim[3])
        self.to(device)

    def forward(self, state):
        # out = F.relu(self.hid_layer_1(state))
        # out = F.relu(self.hid_layer_2(out))
        out = torch.tanh(self.hid_layer_1(state))
        out = torch.tanh(self.hid_layer_2(out))
        out = self.hid_layer_3(out).squeeze()

        return out

class GraphCNNLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 device):
        super().__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.biases = nn.Parameter(torch.FloatTensor(out_dim))
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.biases)
        self.to(device)

    def forward(self, x, adjacent_matrix):
        batch_size, _, _ = x.shape
        out = torch.bmm(x, self.weights.repeat(batch_size, 1, 1))
        out = torch.bmm(adjacent_matrix.repeat(batch_size, 1, 1), out) + self.biases

        return out


class GraphCNN(nn.Module):
    def __init__(self,
                 layer_dim,
                 device):
        super().__init__()
        self.gcn_layer_1 = GraphCNNLayer(layer_dim[0], layer_dim[1], device)
        self.gcn_layer_2 = GraphCNNLayer(layer_dim[1], layer_dim[2], device)
        self.to(device)

    def forward(self, x, adjacent_matrix):
        out = F.leaky_relu(self.gcn_layer_1(x, adjacent_matrix))
        out = F.leaky_relu(self.gcn_layer_2(out, adjacent_matrix))

        return out
    
class lstmNetwork(nn.Module): 
    """
        h_t = relu( fc(h_t_prev) + fc(g_t))
        input_size: input size of the rnn
        hidden_size: hidden size of the rnn(256)
        g_t: 2D tensor of shape (B, hidden_size). Returned from glimpse network.
        h_prev: 2D tensor of shape (B, hidden_size). Hidden state for previous timestep.
        h_t: 2D tensor of shape (B, hidden_size). Hidden state for current timestep.
    """
    
    def __init__(self, hidden_size, device):
        super().__init__()
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.to(device)

    def forward(self, x_t, h_prev, c_prev): 
        h_t, c_t = self.lstm(x_t, (h_prev.detach(), c_prev.detach()))
        
        return h_t, c_t
    
class EGAT(nn.Module):
    def __init__(self, 
                 nfeat,
                 ef_sz, 
                 nhid,
                 nclass, 
                 dropout, 
                 alpha, 
                 nheads,
                 device):
        """
        Dense version of GAT.
        nfeat输入节点的特征向量长度，标量
        ef_sz输入edge特征矩阵的大小，列表，PxNxN
        nhid隐藏节点的特征向量长度，标量
        nclass输出节点的特征向量长度，标量
        dropout：drpout的概率
        alpha：leakyrelu的第三象限斜率
        nheads：attention_head的个数
        """
        super(EGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid*nheads*ef_sz[0], nclass, dropout=dropout, alpha=alpha, concat=False)
        self.to(device)
        
    def forward(self, x, edge_attr):
        x = F.dropout(x, self.dropout, training=self.training)
        temp_x=[]
        for att in self.attentions:
            inn_x,edge_attr=att(x, edge_attr)
            temp_x.append(inn_x)
        x = torch.cat(temp_x, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, edge_attr))
        return F.log_softmax(x, dim=1)