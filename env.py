import numpy as np
import torch
from graph import Graph
import scipy.sparse as sp
from utils import DSN

class Env:
    def __init__(self, data):
        self.data = data
        self.graph = Graph(data)
        self.src = None
        self.dst = None
        self.visited_node = []
        self.visited_edge = []
        self.action = None
        self.success_rate = 0.0
        self.total_reward = 0.0

    def get_info(self, idx):
        self.src, self.dst = self.data.flow_info[idx]
        self.visited_node.append(self.src)

    def normalize_features(self, mx):
        """Row-normalize sparse matrix"""
        """input is a numpy array""" 
        rowsum = mx.sum(axis=1)
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    
    # def get_state(self):
    #     edge_num = len(self.graph.edges)
    #     state = np.zeros([edge_num, 6])

    #     for edge in self.graph.edges.values():
    #         state[edge.idx, 0] = 1 if edge.end_node.idx == self.dst else 0
    #         state[edge.idx, 1] = 0 if edge.idx in self.visited_edge else 1
    #         state[edge.idx, 2] = 1 if edge.idx in self.find_valid_edge() else 0
    #         state[edge.idx, 3] = self.graph.edge_dist_matrix[edge.idx][self.dst]
    #         state[edge.idx, 4] = 1 if edge.idx == self.action else 0
    #         state[edge.idx, 5] = self.graph.edges[edge.idx].slot_num / 10

    #     return torch.from_numpy(state).float()
    def get_state(self):
        edge_num = len(self.graph.edges)
        node_num = len(self.graph.nodes)
        edge_feature = np.zeros([edge_num, 2])
        node_feature = np.zeros([node_num, 4])
        
        for edge in self.graph.edges.values():
            # edge_feature[edge.idx, 0] = 1 if edge.end_node.idx == self.dst else 0
            # edge_feature[edge.idx, 1] = 0 if edge.idx in self.visited_edge else 1
            edge_feature[edge.idx, 0] = 1 if edge.idx in self.find_valid_edge() else 0
            # state[edge.idx, 3] = self.graph.edge_dist_matrix[edge.idx][self.dst]
            # state[edge.idx, 4] = 1 if edge.idx == self.action else 0
            edge_feature[edge.idx, 1] = self.graph.edges[edge.idx].slot_num / 10
            # edge_feature[edge.idx, 3] = len(self.graph.edges[edge.idx].slot_status) / (1024*64)        
        
        for node in self.graph.nodes.values():
            node_feature[node.idx, 0] = 1 if node.idx == self.dst else 0
            node_feature[node.idx, 1] = 0 if node.idx in self.visited_node else 1            
            node_feature[node.idx, 2] = 1 if node.idx in self.find_valid_node() else 0
            node_feature[node.idx, 3] = self.graph.nodes[node.idx].buffer_size
        
        adj = sp.coo_matrix((np.ones(edge_feature.shape[0]), (edge_feature[:, 0], edge_feature[:, 1])), shape=(node_num, node_num), dtype=np.float32)
        node_feature = self.normalize_features(node_feature)
        adj = adj + sp.eye(adj.shape[0])
        
        adj=torch.FloatTensor(np.array(adj.todense()))
        edge_attr =[adj,adj.t(),adj+adj.t()]
        edge_attr=torch.stack(edge_attr,dim=0)
        edge_attr=DSN(edge_attr)
        
        node_feature = torch.FloatTensor(node_feature)
        
        # return torch.from_numpy(edge_feature).float()
        return node_feature,edge_attr
    def convert_to_edge(self,node):
        for i in range(len(self.graph.edges)):
            if self.graph.edges[i].end_node.idx == node and self.graph.edges[i].start_node.idx == self.src:
                edge = self.graph.edges[i].idx
                break
        return  edge
        
    def step(self, action):

        self.action = action
        self.src = self.graph.edges[action].end_node.idx

        # if not self.find_valid_edge():
        #     done = -1
        #     reward = 0
        # elif self.src == self.dst:
        #     done = 1
        #     reward = 5
        #     self.success_rate += 1
        # else:
        #     done = 0
        #     reward = 3 * (1 / self.graph.edge_dist_matrix[action][self.dst])
        #     self.visited_node.append(self.src)
        #     self.visited_edge.append(action)


        
        if self.src == self.dst:
            done = 1
            reward = 5
            self.success_rate += 1
        elif self.src != self.dst and self.find_valid_node():
            done = 0
            reward = 3 * (1 / self.graph.edge_dist_matrix[action][self.dst])
            self.visited_node.append(self.src)
            self.visited_edge.append(action)
        else:
            done = -1
            reward = 0
        self.total_reward += reward
        
        return done, torch.FloatTensor([reward]), self.get_state()

    def find_valid_edge(self):
        valid_edge = []
        for edge in self.graph.edges.values():
            if edge.start_node.idx == self.src \
                    and edge.end_node.idx not in self.visited_node \
                    and edge.slot_num > 0:
                valid_edge.append(edge.idx)

        return valid_edge
    def find_valid_node(self):
        valid_node = []
        node_adj_matrix = self.graph.node_adj_matrix
        valid_matrix = node_adj_matrix - torch.eye(node_adj_matrix.shape[0]).numpy()
        valid_node =[index for (index,value) in enumerate(valid_matrix[self.src]) if value ==1 \
                     and index not in self.visited_node and self.graph.nodes[index].buffer_size > 0] 
        
        return valid_node
    
    def renew(self):
        self.action = None
        self.visited_node.clear()
        self.visited_edge.clear()

    def reset(self):
        self.action = None
        self.visited_node.clear()
        self.visited_edge.clear()

        for edge in self.graph.edges.values():
            edge.slot_num = 10
        for node in self.graph.nodes.values():
            node.buffer_size = 20
        self.action = None
        self.success_rate = 0.0
        self.total_reward = 0.0