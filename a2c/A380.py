import random
import numpy as np


class A380:
    def __init__(self,
                 node_num=12,        # fix node numbers
                 flow_num=100):
        self.node_num = node_num
        self.flow_num = flow_num

        self.node_matrix = []
        self.flow_info = {}
        self.node_info = {}
        self.flow_prd = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.flow_len = [1,16]
        self.flow_delay = [1024, 4096]

    def generate_node_matrix(self):
        self.node_matrix = np.zeros((self.node_num, self.node_num))
        
        # links = [[0, 1], [1, 3], [3, 5], [5, 7], [7, 9], [9, 8], [8, 6], [6, 4],[6, 7], [4, 5], [5, 10],
        #           [10, 6], [10, 7], [10, 4],[4, 2],[2, 0] ,[4, 11], [11, 3],[11, 5],[11, 2],[2, 3]]
        links = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 5], [4, 6], [5, 7], [6, 7], [6, 8],
                  [7, 9], [8, 9], [10, 2], [10, 3], [10, 4], [10, 5], [11, 4], [11, 5], [11, 6], [11, 7]]
        for link in links:
            self.node_matrix[link[0], link[1]] = 1
            self.node_matrix[link[1], link[0]] = 1

    def generate_flow_info(self):
        flow_prd = self.flow_prd
        flow_len = self.flow_len
        flow_delay = self.flow_delay
        for idx in range(self.flow_num):
            src = random.randint(0, self.node_num - 1)
            dst = random.randint(0, self.node_num - 1)
            while src == dst:
                dst = random.randint(0, self.node_num - 1)
            prd = flow_prd[random.randint(0, len(flow_prd) - 1)]
            length = random.randint(flow_len[0], flow_len[1])
            delay = random.randint(flow_delay[0], flow_delay[1])
            
            self.flow_info[idx] = [src, dst, length, prd, delay]
        # self.flow_info = dict(sorted(self.flow_info.items(), key = lambda x: x[1][2],reverse=True))
        
        # record=np.zeros(16)
        # for i in range(16):
        #     for j in self.flow_info :
        #         if self.flow_info[j][2]==i+1:
        #             record[i]+=1
        # record=list(reversed(record))
        # r=0
        # tmp = []
        # sorted_flow={}
        # for i in range(len(record)):
        #     l = int(record[i])
        #     if l>1:
        #         # s=[i for i in list(self.flow_info.keys())[r:r+l]]
        #         tmp = dict(list(self.flow_info.items())[r:r+l])
        #         tmp = dict(sorted(tmp.items(), key=lambda x: abs(x[1][0]-x[1][1]),reverse=True))
        #         # print(tmp)
        #         sorted_flow.update(tmp)
        #         r+=l
        #     else:
        #         r+=l
        #         continue
        # self.flow_info = sorted_flow
        
    def generate_all_data(self):
        self.generate_node_matrix()
        self.generate_flow_info()