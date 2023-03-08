import torch
import torch.nn as nn
import torch.optim as optim
from network import EdgeBatchNorm, QNetwork, EGAT, QNetwork_conv


class Model(nn.Module):
    def __init__(self,
                 lr,
                 device):
        super().__init__()
        self.edge_bn = EdgeBatchNorm(dim=4,
                                     device=device)
        # self.q_net = QNetwork(layer_dim=[16, 16, 4, 1],
        #                       device=device)
        self.q_net_conv = QNetwork_conv(layer_dim=[16, 1],
                                        device=device)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=5e-4)
        
        self.egat = EGAT(nfeat = 4,             # args.state_dim,             
                        ef_sz = (3,12,12),      # tuple(edge_attr.shape)   
                        nhid  = 16, 
                        nclass = 16,             # output size 
                        dropout = 0.6, 
                        nheads = 4, 
                        alpha = 0.2,
                        device = device)
                        
    
    def forward(self, state):
        # out = self.edge_bn(state)
        # out = self.q_net(state)
        out = self.q_net_conv(state)
        return out

    # def valid_choice(self, state, mask):
    #     out = self.forward(state)
    #     valid_choice = torch.take(out.cpu(), torch.LongTensor(mask))

    #     return valid_choice
    
    def valid_choice(self, node_feature, edge_attr, mask):
        state = self.egat(node_feature, edge_attr)
        # state = state.unsqueeze(dim=2)
        # state = state.unsqueeze(dim=3)
        state = state.view(1,state.size()[0],state.size()[1],1).permute(0,2,1,3)
        out = self.forward(state)
        out = out.reshape(12,-1)
        valid_choice = torch.take(out.cpu(), torch.LongTensor(mask))

        return valid_choice