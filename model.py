import torch.nn as nn
from network import GraphCNN, LinkBatchNorm, QNetwork,GAT
from param import args

class Model(nn.Module):
    def __init__(self,
                 gcn_layer_dim,
                 q_layer_dim,
                 device):
        super().__init__()
        self.link_bn = LinkBatchNorm(dim=gcn_layer_dim[0],
                                     device=device)
        self.gcn = GraphCNN(layer_dim=gcn_layer_dim,
                            device=device)
        self.gat = GAT(nfeat = 4,           #in feature nums
                       nhid = args.hidden, 
                       nclass = 64,         # output size 
                       dropout = args.dropout, 
                       nheads = args.nb_heads, 
                       alpha = args.gat_alpha,
                       device = device)
        self.qnet = QNetwork(layer_dim=q_layer_dim,
                             device=device)

    def forward(self, state, adjacent_matrix):
        # embed_state = self.gcn(self.link_bn(state), adjacent_matrix)
        embed_state = self.gat(state, adjacent_matrix)  #!!!
        out = self.qnet(embed_state)

        return out