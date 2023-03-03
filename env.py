import numpy as np
import torch
from graph import Graph


class Env:
    def __init__(self, data):
        self.data = data
        self.graph = Graph(data)
        self.src = None
        self.dst = None
        self.visited_node = []
        self.visited_edge = []
        self.action = None
        self.success_num = 0
        self.total_reward = 0.0
        self.accumulated_delay = 0.0
        self.delay = 0.0
        self.count_delay = 0
        self.count_full = 0
    def get_info(self, idx):
        self.src, self.dst, self.flow_len, self.flow_prd, self.flow_delay = self.data.flow_info[idx]
        self.visited_node.append(self.src)

    def get_state(self):
        edge_num = len(self.graph.edges)
        node_num = len(self.graph.nodes)
        state = np.zeros([edge_num, 4])
        node_feature = np.zeros([node_num, 4])
        
        for edge in self.graph.edges.values():
            state[edge.idx, 0] = 1 if edge.end_node.idx == self.dst else 0
            state[edge.idx, 1] = 0 if edge.idx in self.visited_edge else 1
            state[edge.idx, 2] = 1 if edge.idx in self.find_valid_edge() else 0
            # state[edge.idx, 3] = self.graph.edge_dist_matrix[edge.idx][self.dst]
            # state[edge.idx, 4] = 1 if edge.idx == self.action else 0
            # state[edge.idx, 5] = self.graph.edges[edge.idx].slot_num / 10
            state[edge.idx, 3] = len(self.graph.edges[edge.idx].slot_status) / (1024*64)        
        
        for node in self.graph.nodes.values():
            node_feature[node.idx, 0] = 1 if node.idx == self.dst else 0
            node_feature[node.idx, 1] = 0 if node.idx in self.visited_node else 1            
            node_feature[node.idx, 2] = 1 if node.idx in self.find_valid_node() else 0
            node_feature[node.idx, 3] = self.graph.nodes[node.idx].buffer_size
        
        return torch.from_numpy(state).float()
    def step(self, action, offset):
        self.action = action
        self.src = self.graph.edges[action].end_node.idx
        self.accumulated_delay += self.calculate_delay(offset)
        if self.accumulated_delay >= self.flow_delay:
            self.count_delay+=1
        if not self.find_valid_edge() or not offset:
            self.count_full += 1
            
        if self.src == self.dst and self.accumulated_delay <= self.flow_delay:
            done = 1
            reward = 10
            self.success_num += 1
            
        elif self.src != self.dst and self.find_valid_edge() and self.accumulated_delay <= self.flow_delay and offset:
            done = 0
            reward = 1 / self.graph.edge_dist_matrix[action][self.dst]
            self.visited_node.append(self.src)
            self.visited_edge.append(action)
        else:
            done = -1
            reward = 0

        self.total_reward += reward

        return done, torch.FloatTensor([reward]), self.get_state()
    
    # def step(self, action, offset):
    #     self.action = action
    #     self.src = self.graph.edges[action].end_node.idx

    #     if offset:
    #         self.delay += offset[0]

    #     if self.src == self.dst and self.delay <= self.flow_delay and offset:
    #         done = 1
    #         reward = 10
    #         self.success_num += 1
    #     elif self.src != self.dst and self.find_valid_edge() and self.delay <= self.flow_delay and offset:
    #         done = 0
    #         reward = 1 / self.graph.edge_dist_matrix[action][self.dst] - (self.delay / 6400)
    #         self.visited_node.append(self.src)
    #         self.visited_edge.append(action)
    #     else:
    #         done = -1
    #         reward = -10
    #         if self.find_valid_edge() and self.delay > self.flow_delay:
    #             reward = -5
    #         elif self.find_valid_edge() and not offset:
    #             reward = -5

    #     self.total_reward += reward

    #     return done, torch.FloatTensor([reward]), self.get_state()
    
    def find_valid_edge(self):
        valid_edge = []
        for edge in self.graph.edges.values():
            if edge.start_node.idx == self.src \
                    and edge.end_node.idx not in self.visited_node :
                valid_edge.append(edge.idx)

        return valid_edge

    def find_valid_node(self):
        valid_node = []
        node_adj_matrix = self.graph.node_adj_matrix
        valid_matrix = node_adj_matrix - torch.eye(node_adj_matrix.shape[0]).numpy()
        valid_node =[index for (index,value) in enumerate(valid_matrix[self.src]) if value ==1 and index not in self.visited_node] 
        
        return valid_node
    
    def find_slot(self, link_idx):
        return self.graph.edges[link_idx].find_slot(flow_length = self.flow_len, flow_prd = self.flow_prd)
    
    def occupy_slot(self, action):
        link_idx, offset = action
        self.graph.edges[link_idx].occupy_slot(offset)
    
    @staticmethod
    def calculate_delay(position):
        return position[0] if position else 64      # 64 :slot per phase
    
    def renew(self):
        self.action = None
        self.visited_node.clear()
        self.visited_edge.clear()
        self.accumulated_delay = 0.0
        
    def reset(self):
        self.action = None
        self.visited_node.clear()
        self.visited_edge.clear()
        for edge in self.graph.edges.values():
            edge.slot_status.clear()
            # edge.slot_status = [i for i in range(64*1024)]
            edge.slot_status = [1 for _ in range(1024*64)]
        # for edge in self.graph.edges.values():
        #     edge.slot_num = 10
        self.count_delay = 0
        self.count_full = 0
        self.action = None
        self.success_num = 0
        self.total_reward = 0.0