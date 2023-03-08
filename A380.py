import random
import numpy as np


class A380:
    def __init__(self,
                 node_num=12,
                 flow_num=20):
        self.node_num = node_num
        self.flow_num = flow_num

        self.node_matrix = []
        self.flow_info = {}

        self.generate_all_data()

    def generate_node_matrix(self):
        self.node_matrix = np.zeros((self.node_num, self.node_num))
        
        # links = [[0, 1], [1, 3], [3, 5], [5, 7], [7, 9], [9, 8], [8, 6], [6, 4],[6, 7], [4, 5], [5, 10],
        #           [10, 6], [10, 7], [10, 4],[4, 2],[2, 0] ,[4, 11], [11, 3],[11, 5],[11, 2],[2, 3]]
        links = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 5], [4, 6], [5, 7], [6, 7], [6, 8],
                  [7, 9], [8, 9], [10, 2], [10, 3], [10, 4], [10, 5], [11, 4], [11, 5], [11, 6], [11, 7]]
        for link in links:
            self.node_matrix[link[0], link[1]] = 1
            self.node_matrix[link[1], link[0]] = 1
        print(self.node_matrix)
        
    def generate_flow_info(self):
        for idx in range(self.flow_num):
            src = random.randint(0, self.node_num - 1)
            dst = random.randint(0, self.node_num - 1)
            while src == dst:
                dst = random.randint(0, self.node_num - 1)

            self.flow_info[idx] = [src, dst]

    def generate_all_data(self):
        self.generate_node_matrix()
        self.generate_flow_info()

# if __name__ == '__main__':
#     t = A380()
#     t.generate_node_matrix()