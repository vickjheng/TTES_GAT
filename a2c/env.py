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

    def get_info(self, idx):
        self.src, self.dst = self.data.flow_info[idx]
        self.visited_node.append(self.src)

    def get_state(self):
        edge_num = len(self.graph.edges)
        state = np.zeros([edge_num, 6])

        for edge in self.graph.edges.values():
            state[edge.idx, 0] = 1 if edge.end_node.idx == self.dst else 0
            state[edge.idx, 1] = 0 if edge.idx in self.visited_edge else 1
            state[edge.idx, 2] = 1 if edge.idx in self.find_valid_edge() else 0
            state[edge.idx, 3] = self.graph.edge_dist_matrix[edge.idx][self.dst]
            state[edge.idx, 4] = 1 if edge.idx == self.action else 0
            state[edge.idx, 5] = self.graph.edges[edge.idx].slot_num / 10

        return torch.from_numpy(state).float()

    def step(self, action):
        self.action = action
        self.src = self.graph.edges[action].end_node.idx

        if self.src == self.dst:
            done = 1
            reward = 10
            self.success_num += 1
        elif self.src != self.dst and self.find_valid_edge():
            done = 0
            reward = 1 / self.graph.edge_dist_matrix[action][self.dst]
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
        self.action = None
        self.success_num = 0
        self.total_reward = 0.0