import os
import shutil
import time
import datetime
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ladder import Ladder
from env import Env
from memory import Memory
from model import Model
from param import args


class Trainer:
    def __init__(self):
        if args.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.data = Ladder()
        self.data.read_file()
        self.env = Env(self.data)
        self.memory = Memory(args.capacity)
        self.eval_q = Model(gcn_layer_dim=args.gcn_layer_dim,
                            q_layer_dim=args.q_layer_dim,
                            device=self.device)
        # self.eval_q = Model()
        #TODO (self, in_features, out_features, dropout=args.dropout, alpha=args.gat_alpha, concat=True):
        self.target_q = deepcopy(self.eval_q)
        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = optim.SGD(self.eval_q.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=args.lr_decay)
        self.epsilon = args.epsilon
        self.env_record = {'success_rate': [], 'usage': [], 'reward': []}
        self.training_record = []
        self.scalar = []

    @torch.no_grad()
    def generate_experience(self):
        print(datetime.datetime.now().strftime('\n[%m-%d %H:%M:%S]'),
              '---------- Generating experience')
        self.env.reset()
        flow_indices = [idx for idx in range(len(self.data.flow_info))]
        random.shuffle(flow_indices)

        for flow_idx in flow_indices:
            action_transition = []
            self.env.flow_info(flow_idx)
            state = self.env.get_state()
            while True:
                adjacent_matrix = torch.from_numpy(self.env.graph.link_adjacent_matrix).unsqueeze(dim=0).float().to(self.device)
                # adjacent_matrix = torch.from_numpy(self.env.graph.link_adjacent_matrix).float().to(self.device)
                link_q = self.eval_q(state.unsqueeze(dim=0).to(self.device), adjacent_matrix).reshape(-1)
                # link_q = self.eval_q((state).to(self.device), adjacent_matrix).reshape(-1)               
                link_idx = self.select_action(link_q)
                offset = self.env.find_slot(link_idx)

                current_state = state
                mask = self.env.find_valid_link()
                done, reward, state = self.env.update([link_idx, offset])
                # print(reward)

                action_transition.append([link_idx, offset])
                self.memory.push(current_state, link_idx, reward, state, mask)  # store (s, a, r, s', mask)

                if done == 1:                                                   # successfully scheduled
                    for action in action_transition:
                        self.env.occupy_slot(action)
                    break
                elif done == -1:                                                # unsuccessfully scheduled
                    break
                elif done == 0:                                                 # in progress
                    self.env.renew()

            self.env.refresh()

        success_rate = self.env.success_rate(len(flow_indices))
        usage = self.env.usage()
        reward = self.env.mean_reward(len(flow_indices))
        self.env_record['success_rate'].append(success_rate)
        self.env_record['usage'].append(usage)
        self.env_record['reward'].append(reward)
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Success Rate: {:.2f}% |'.format(success_rate * 100),
              'Usage: {:.2f}% |'.format(usage * 100),
              'Reward: {:.2f}'.format(reward))

    # select action through epsilon-greedy method
    def select_action(self, link_q):
        if random.random() < self.epsilon:                                # exploration
            link_idx = random.choice(self.env.find_valid_link())
        else:                                                             # exploitation
            link_idx = self.env.find_valid_link()[torch.argmax(torch.take(link_q.cpu(), torch.LongTensor(self.env.find_valid_link()))).item()]

        return link_idx

    def train_one_episode(self):
        print(datetime.datetime.now().strftime('\n[%m-%d %H:%M:%S]'),
              '---------- Training memory')
        self.eval_q.train()
        #!!!
        if len(self.memory) < args.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(args.batch_size, self.device)

        # update eval_q
        self.eval_q.zero_grad()
        adjacent_matrix = torch.from_numpy(self.env.graph.link_adjacent_matrix).unsqueeze(dim=0).float().to(self.device)
        current_q = self.eval_q(state_batch, adjacent_matrix).squeeze().gather(1, action_batch).reshape(-1)
        target_q = self.target_q(next_state_batch.detach(), adjacent_matrix.detach()).reshape(args.batch_size, -1)
        max_q = []
        for q, mask in zip(target_q, mask_batch):
            max_q.append(torch.max(torch.take(q, torch.LongTensor(mask).to(self.device))))
        target_q = (reward_batch + args.gamma * torch.stack(max_q)).detach()

        # for idx in range(len(current_q)):
        #     print('{:+.4f}/{:+.4f}'.format(current_q[idx].item(), target_q[idx].item()))

        loss = self.loss_function(current_q, target_q)
        loss.backward()
        self.optimizer.step()

        self.training_record.append(loss.item())
        self.scalar.append(loss)
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Q Value: {:.4f}/{:.4f} |'.format(torch.mean(current_q), torch.mean(target_q)),
              'Loss: {:.4f} |'.format(loss), end=' ')

    def train(self):
        for episode in range(args.episodes):
            episode += 1
            start_time = time.time()
            print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                  'Episode: {:04d}'.format(episode))
            print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                  'Epsilon: {:.2f}'.format(self.epsilon))

            self.generate_experience()
            self.train_one_episode()
            print('Time: {:.2f}s'.format(time.time() - start_time))

            # if episode > args.lr_decay_start_episode:                                    # lr decay after exploration
            #     self.scheduler.step()
            if self.epsilon > args.end_epsilon:  # epsilon decay while exploration
                self.epsilon *= args.epsilon_decay
            if episode % args.update_target_q_step == 0:                                    # copy parameters to target_q
                self.target_q.load_state_dict(self.eval_q.state_dict())
                print(datetime.datetime.now().strftime('\n[%m-%d %H:%M:%S]'),
                      '---------- Copying parameters')
            if episode % args.save_record_step == 0:                                        # saving record for drawing
                self.save_record(episode)
                print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                      '---------- Saving record')
            print('#' * 70)

    def save_record(self, episode):
        if not os.path.exists('record/env'):
            os.makedirs('record/env')
        for key in self.env_record.keys():
            np.save(f'record/env/{key}_{episode}.npy', self.env_record[key])
        np.save(f'record/loss_{episode}.npy', self.training_record)


def main():
    if os.path.exists('record'):
        shutil.rmtree('record')
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()