import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkBatchNorm(nn.Module):
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
    
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, 
                 in_features, 
                 out_features,
                 dropout, 
                 alpha,
                 device,
                 concat=True):
        
        super(GraphAttentionLayer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.to(device)
        
    def forward(self, h, adj):
        batch_size,_,_ = h.shape 
        # print(f'---------------\nh.size: {h.size()}\nself.W.size: {self.W.size()}')
        Wh = torch.bmm(h, self.W.repeat(batch_size, 1, 1)) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        # e = Wh1 + Wh2.T
        e = Wh1 + Wh2.transpose(1,2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, 
                 nfeat,
                 nhid,
                 nclass,
                 dropout,
                 alpha,
                 nheads,
                 device):
        
        """Dense version of GAT."""
        super(GAT, self).__init__()
        
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True,device = device) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, device = device,concat=False)
                                         #( 8*8,64)
        self.outsize = nhid * nheads
        self.to(device)
        
    def forward(self, x, adj):
        batch_size,states_num,_ = x.shape
        x = F.dropout(x, self.dropout, training=self.training)
        peep = [att(x, adj) for att in self.attentions]
        x = torch.cat(peep, -1) #XXX
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
    
    
class QNetwork(nn.Module):
    def __init__(self,
                 layer_dim,
                 device):
        super().__init__()
        self.hid_layer_1 = nn.Linear(layer_dim[0], layer_dim[1])
        self.hid_layer_2 = nn.Linear(layer_dim[1], layer_dim[2])
        self.hid_layer_3 = nn.Linear(layer_dim[2], layer_dim[3])
        self.to(device)

    def forward(self, embed_state):
        out = F.relu(self.hid_layer_1(embed_state))
        out = F.relu(self.hid_layer_2(out))
        out = self.hid_layer_3(out)

        return out
