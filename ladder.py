import random
import numpy as np


class Ladder:
    def __init__(self,
                 node_num=12,
                 flow_num=100):
        self.node_num = node_num
        self.flow_num = flow_num

        self.node_matrix = []
        self.flow_info = {}

        self.generate_all_data()

    def generate_node_matrix(self):
        self.node_matrix = np.zeros((self.node_num, self.node_num))
        links = []
        idx = 0

        while idx + 1 < self.node_num:
            links.append([idx, idx + 1])
            idx += 2
        idx = 0

        while idx + 2 < self.node_num:
            links.append([idx, idx + 2])
            idx += 1

        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    self.node_matrix[i, j] = 1

        for link in links:
            self.node_matrix[link[0], link[1]] = 1
            self.node_matrix[link[1], link[0]] = 1

    def generate_flow_info(self):
        flow_prd = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        flow_len = [1,16]
        flow_delay = [1024, 4096]
        for idx in range(self.flow_num):
            src = random.randint(0, self.node_num - 1)
            dst = random.randint(0, self.node_num - 1)
            while src == dst:
                dst = random.randint(0, self.node_num - 1)
            prd = flow_prd[random.randint(0, len(flow_prd) - 1)]
            length = random.randint(flow_len[0], flow_len[1])
            delay = random.randint(flow_delay[0], flow_delay[1])
            
            self.flow_info[idx] = [src, dst, length, prd, delay]

    def generate_all_data(self):
        self.generate_node_matrix()
        self.generate_flow_info()