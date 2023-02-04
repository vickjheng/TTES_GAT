import os
import shutil
import datetime
import random
import numpy as np
import torch
from ladder import Ladder
from param import args
from model_save_16units import *
from env import Env
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self,
                 data):
        if args.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.data = data
        # self.data.generate_all_data()
        self.data.read_file()
        self.vld_record = {'reward': [], 'done': []}
        self.eval_q = torch.load('./model_save_32units/model_5000.pt')
        self.env = Env(self.data)
        # for name, param in self.eval_q.named_parameters():
        #     print(name, ':', param.requires_grad)
    def train(self):
        for episode in range(args.episodes):
            episode += 1
            print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                  'Episode: {:04d}'.format(episode))

            print(datetime.datetime.now().strftime('\n[%m-%d %H:%M:%S]'),
                  '---------- Collecting experience')
            reward = self.collect_experience()
            print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                  'Reward: {:.4f} |'.format(reward))

            print(datetime.datetime.now().strftime('\n[%m-%d %H:%M:%S]'),
                  '---------- Training')
            loss = self.learn()
            print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                  'Loss: {:.4f} |'.format(loss.item()))

            self.record['reward'].append(reward)
            self.record['loss'].append(loss.item())
            if episode % args.save_record_step == 0:
                self.save_record(episode)
            print('#' * 70)

    def save_record(self, episode):
        if not os.path.exists('record/env'):
            os.makedirs('record/env')
        for key in self.record.keys():
            np.save(f'record/env/{key}_{episode}.npy', self.record[key])
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              '---------- Saving record')
        
        # select action through epsilon-greedy method
    def select_action(self, link_q):

        link_idx = self.env.find_valid_link()[torch.argmax(torch.take(link_q.cpu(), torch.LongTensor(self.env.find_valid_link()))).item()]

        return link_idx
    
    @torch.no_grad()
    def validate(self, data):
        self.eval_q.eval()

        self.env.reset()
        flow_indices = [idx for idx in range(len(data.flow_info))]

        for flow_idx in flow_indices:
            action_transition = []
            self.env.flow_info(flow_idx)
            state = self.env.get_state()
            while True:
                adjacent_matrix = \
                    torch.from_numpy(self.env.graph.link_adjacent_matrix).unsqueeze(dim=0).float().to(self.device)
                link_q = self.eval_q(state.unsqueeze(dim=0).to(self.device), adjacent_matrix).reshape(-1)
                link_idx = self.select_action(link_q)
                offset = self.env.find_slot(link_idx)

                current_state = state
                mask = self.env.find_valid_link()
                done, reward, state ,terminal= self.env.update([link_idx, offset])
                # is_terminal = 0 if done == 0 else 1
                action_transition.append([link_idx, offset])
                # self.store_transition(current_state, link_idx, reward, state, is_terminal, mask)

                if done == 1:
                    for action in action_transition:
                        self.env.occupy_slot(action)
                    break
                elif done == -1:
                    break
                elif done == 0:
                    self.env.renew()

            self.env.refresh()
        
        success_rate = self.env.success_rate(len(flow_indices))
        reward = self.env.mean_reward(len(flow_indices))
        # self.vld_record['reward'].append(reward)
        # self.vld_record['done'].append(success_rate)
        
        return reward,success_rate
#TODO mov_avg for numerous data
def draw(datas):
    fig, plot = plt.subplots(len(datas))
    y = datas
    # y = move_average(np.load(f'history/0111_3/loss_{episode}.npy'))
    x = np.arange(len(y['reward']))
    for idx, key in zip(range(len(datas)), datas.keys()):
        mean = np.mean(datas[key])
        print(f'{key} mean: {mean}')
        plot[idx].bar(x, datas[key])
        plot[idx].axhline(mean,color='r',label='mean')
        
        # plot[idx].set_xlim(int(window_size / 2), int(episode - (window_size / 2)))
        # plot[idx].grid(True)
        plot[idx].set_title(f'{key}')
    plt.legend(loc='best')    
    fig.tight_layout()
    
# datas = {'reward': [-2,-3,-5,3], 'done': [0,0,1,1]}
# print(np.mean(datas['reward']))

# plt.figure()
# x = np.arange(len(datas['reward']))
# plt.plot(x, datas['reward'], 'b')
# mean = np.mean(datas['reward'])
# plt.axhline(mean,color='r',label='mean')
# plt.legend(loc='best')
# plt.grid(True)
# plt.show()

def main():
    # trainer.train()

    times = 100
    vld_record = {'reward': [], 'success_rate': []}
    for time in range(times):
        node_num = random.randint(8, 12)
        flow_num = random.randint(10, 20)
        data = Ladder(node_num=node_num,
                      flow_num=flow_num)
        data.generate_all_data(flow_len = args.flow_len, flow_prd = args.flow_prd, flow_delay = args.flow_delay)
        data.read_file()
        trainer = Trainer(data)
        
        reward,success_rate = trainer.validate(data=data)

        vld_record['reward'].append(reward)
        vld_record['success_rate'].append(success_rate)
        
        print('{:+.4f}'.format(reward))
    draw(vld_record)

if __name__ == '__main__':
    main()

    