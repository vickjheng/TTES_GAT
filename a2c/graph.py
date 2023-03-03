import numpy as np


class Node:
    def __init__(self, idx):
        self.idx = idx
        # self.end_node = end_node
        self.buffer_size = 20
        
class Edge:
    def __init__(self, idx, start_node, end_node):
        self.idx = idx
        self.start_node = start_node
        self.end_node = end_node
        # self.slot_num = 15
        self.slot_per_phase = 64
        self.hyper_period = 1024
        # self.slot_status = [i for i in range(self.hyper_priod * self.slot_per_phase)]
        self.slot_status = [1 for _ in range(self.hyper_period * self.slot_per_phase)]
        
    def find_slot(self, flow_length, flow_prd):
        frames = int(self.hyper_period / flow_prd)

        for position in range(self.slot_per_phase - flow_length):
            offset = []
            temp = [position + length for length in range(flow_length)]
            flag = 0
            for frame in range(frames):
                if frame > 0:
                    for idx in range(len(temp)):
                        temp[idx] += flow_prd * self.slot_per_phase
                offset.extend(temp)

                for idx in temp:
                    if self.slot_status[idx] == 1:
                        flag += 1
                    else:
                        break

                if flag == flow_length * (frame + 1):
                    continue
                else:
                    break

            if flag == flow_length * frames:
                return offset

        return []
    
    def occupy_slot(self, offset):
        for idx in offset:
            self.slot_status[idx] -= 1

            if self.slot_status[idx] == -1:
                raise ValueError('Slot status value error!')
                
    # def find_slot(self, flow_len, flow_prd):
    #     for offset in range(self.slot_per_phase - flow_len):
    #         flag = 0
    #         frames =int(self.hyper_priod/flow_prd)    
    #         positions = []
    #         for i in range(flow_len): 
    #             if flag:
    #                 break
    #             for frame in range(frames):
    #                 pos = (offset+i)+ frame * self.slot_per_phase
    #                 if pos in self.slot_status:
    #                     positions.append(pos)
    #                 else:
    #                     flag = 1
    #                     break
    #         if flag:
    #             continue
    #         else:    

    #             return positions
    #     return []
    # def occupy_slot(self,positions):
    #     for position in positions:
    #         self.slot_status.remove(position)


class Graph:
    def __init__(self, data):
        self.data = data

        self.nodes = {}
        self.node_adj_matrix = []
        self.edges = {}
        self.edge_adj_matrix = []
        self.edge_dist_matrix = []

        self.load_nodes()
        self.load_node_adj_matrix()
        self.get_edges()
        self.get_edge_dist_matrix()

    def load_nodes(self):
        self.nodes = {}
        node_num = len(self.data.node_matrix)
        for idx in range(node_num):
            self.nodes[idx] = Node(int(idx))

    def load_node_adj_matrix(self):
        self.node_adj_matrix = self.data.node_matrix

    def get_edges(self):
        self.edges = {}
        node_num = len(self.node_adj_matrix)
        start_from_node = {}
        for idx in range(node_num):
            start_from_node[idx] = []
        idx = 0
        for i in range(node_num):
            for j in range(node_num):
                if self.node_adj_matrix[i][j] == 0 or i == j:
                    continue
                self.edges[idx] = Edge(idx, self.nodes[i], self.nodes[j])
                start_from_node[i].append(idx)
                idx += 1
        edge_num = idx
        self.edge_adj_matrix = np.zeros([edge_num, edge_num])
        for i in range(edge_num):
            for j in start_from_node[self.edges[i].end_node.idx]:
                self.edge_adj_matrix[i][j] = 1

    def get_edge_dist_matrix(self):
        node_num = len(self.nodes)
        edge_num = len(self.edges)
        self.edge_dist_matrix = np.full([edge_num, node_num], 999)

        for idx in range(edge_num):
            visited_edges = [idx]
            curr_list = [self.edges[idx]]
            distance = 1
            self.edge_dist_matrix[idx][self.edges[idx].start_node.idx] = 0
            while len(curr_list) > 0:
                next_list = []
                for curr_edge in curr_list:
                    self.edge_dist_matrix[idx][curr_edge.end_node.idx] = \
                        min(self.edge_dist_matrix[idx][curr_edge.end_node.idx], distance)
                    for adj_idx in range(edge_num):
                        if self.edge_adj_matrix[curr_edge.idx][adj_idx] == 1 \
                                and adj_idx not in visited_edges:
                            visited_edges.append(adj_idx)
                            next_list.append(self.edges[adj_idx])
                curr_list = next_list
                distance += 1