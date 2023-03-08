import os
import shutil
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
from ladder import Ladder
from env import Env
from model import Model


class Trainer:
    def __init__(self):
        self.device = torch.device('cuda')
        self.data = Ladder()
        self.data.generate_all_data()
        self.env = Env(self.data)
        self.model = Model(lr=0.0001,
                           device=self.device)
        self.loss_function = nn.SmoothL1Loss()
        self.epsilon = 1.0
        self.batch_size = 32
        self.train_record = {'reward': [], 'loss': [], 'success_rate': []}
        self.val_reward = []
        # self.hidden_state, self.cell_state = self.env.init_states()

    def train_one_episode(self):
        self.model.train()
        self.model.optimizer.zero_grad()

        # self.data.generate_all_data()
        # self.env = Env(self.data)
        self.env.reset()

        loss_record = []

        for idx in range(len(self.data.flow_info)):
            self.env.get_info(idx)
            # state = self.env.get_state()
            node_feature,edge_attr = self.env.get_state()
            action_transition = []
            tmp = []
            check_record = {'flow_info': [],'action': [],'done': []}
            ctrl_gate = 0
            while True:
                check_record['flow_info'].append(self.data.flow_info[idx])
                # mask = self.env.find_valid_edge()
                mask = self.env.find_valid_node()
                
                # actions = \
                #     self.model.valid_choice(state.unsqueeze(dim=0).to(self.device), mask)
                if not mask:
                    break
                if ctrl_gate and len(mask)>1:
                    for choice in mask:
                        edge = self.env.convert_to_edge(choice)
                        tmp.append(self.env.graph.edge_dist_matrix[edge][self.env.dst])
                    pickout = max(tmp)
                    pickout = tmp.index(pickout)
                    # print('pickout: ',pickout)
                    del mask[pickout]
                
                ctrl_gate = 0
                
                actions = self.model.valid_choice(node_feature.to(self.device),
                                                  edge_attr.to(self.device),
                                                  mask)
                
                action, q_value = self.select_action(actions, mask, greedy='False')
                pred_q = q_value
                
                # action : node ---> edge
                node = action
                action = self.env.convert_to_edge(node)
                # print(f'src :{self.env.src}\npick edge :{action}\ndst :{self.env.dst}\n----------')
                if (self.env.src % 2 == 0 and node % 2 != 0) or \
                    (self.env.src % 2 != 0 and node % 2 == 0):
                    ctrl_gate = 1
                # done, reward, state = self.env.step(action)
                done, reward, state = self.env.step(action)
                node_feature ,edge_attr = state
                
                if done == 0:
                    # mask = self.env.find_valid_edge()
                    # actions = \
                    #     self.model.valid_choice(state.unsqueeze(dim=0).to(self.device), mask)
                    mask = self.env.find_valid_node()
                    actions = self.model.valid_choice(node_feature.to(self.device),
                                                      edge_attr.to(self.device),
                                                      mask)
                    next_action, q_value = self.select_action(actions, mask, greedy='True') # True for DQN

                    target_q = reward + 0.9 * q_value
                else:
                    target_q = reward

                # print('{:+.04f}|{:+.04f}'.format(pred_q.item(), target_q.item()))

                loss = self.loss_function(pred_q, target_q)
                loss.backward()
                self.model.optimizer.step()

                self.train_record['loss'].append(loss.item())

                loss_record.append(loss.item())


                action_transition.append([node,action])

                if done == 1:
                    for action in action_transition:
                        node,edge = action
                        self.env.graph.nodes[node].buffer_size -= 1

                        self.env.graph.edges[edge].slot_num -= 1
                    break
                elif done == -1:
                    break
                elif done == 0:
                    continue

            self.env.renew()

        reward = self.env.total_reward / len(self.data.flow_info)
        success_rate = self.env.success_rate / len(self.data.flow_info)
        mean_loss = sum(loss_record) / len(loss_record)
        self.train_record['reward'].append(reward)
        self.train_record['success_rate'].append(success_rate)
        
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Reward: {:+.02f} |'.format(reward),
              'Rate: {:.02f} |'.format(success_rate),
              'Loss: {:.04f}'.format(mean_loss))

    def select_action(self, actions, mask, greedy):
        if greedy == 'False':
            if random.random() < self.epsilon:
                action = random.choice(mask)
                q_value = actions[mask.index(action)]
            else:
                action = mask[torch.argmax(actions).item()]
                q_value = torch.max(actions)

            return action, q_value

        elif greedy == 'True':
            action = mask[torch.argmax(actions).item()]
            q_value = torch.max(actions)

            return action, q_value

    def decrement_epsilon(self):
        epsilon_decay = 0.9999
        min_epsilon = 0.01
        self.epsilon = max(self.epsilon * epsilon_decay, min_epsilon)

    def save_record(self, episode):
        savfile='train_record_conv'
        savmdl='model_save_conv'
        
        # if episode % 100 == 0:
        if not os.path.exists(savfile):
            os.makedirs(savfile)
        for key in self.train_record.keys():
            np.save(f'{savfile}/{key}_{episode}.npy', self.train_record[key])
        if not os.path.exists(savmdl):
            os.makedirs(savmdl)
        
        FILE = f'./{savmdl}/model_{episode}.pt'
        torch.save(self.model, FILE)

    def train(self):
        episode = 0
        while True:
            episode += 1
            print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                  'Episode: {:04d}'.format(episode))
            print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                  'Epsilon: {:.06f}'.format(self.epsilon), end='\n\n')

            self.train_one_episode()
            if episode >300:
                self.decrement_epsilon()
            if episode %100 ==0:
                self.save_record(episode)
            print('#' * 40)


def main():
    record_path = ['train_record_conv', 'val_record_conv']
    for path in record_path:
        if os.path.exists(path):
            shutil.rmtree(path)
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()