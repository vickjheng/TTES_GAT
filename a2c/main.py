import os
import shutil
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ladder import Ladder
from env import Env
from model import Actor, Critic


class Trainer:
    def __init__(self):
        self.device = torch.device('cuda')
        self.data = Ladder()
        self.data.generate_all_data()
        self.env = Env(self.data)
        self.actor = Actor(lr=0.0001,
                           device=self.device)
        self.critic = Critic(lr=0.0001,
                             device=self.device)
        self.loss_function = nn.SmoothL1Loss()
        self.record = {'reward': [], 'rate': [], 'pi_loss': [], 'v_loss': []}

    def train_one_episode(self):
        self.actor.train()
        self.critic.train()

        self.env.reset()

        loss_record = {'pi_loss': [], 'v_loss': []}
        
        check_record = {'flow_info': [],'action': [],'done': []}
        for idx in range(len(self.data.flow_info)):
            
            check_record['flow_info'].append(self.data.flow_info[idx])
            
            self.env.get_info(idx)
            state = self.env.get_state()
            action_transition = []

            while True:
                mask = self.env.find_valid_edge()
                if not mask:
                    break
                # actor
                actions = \
                    self.actor.valid_choice(state.unsqueeze(dim=0).to(self.device), mask)
                action, log_prob = self.select_action(actions, mask)
                check_record['action'].append(action)
                # critic
                value = self.critic(state.unsqueeze(dim=0).to(self.device))

                done, reward, state = self.env.step(action)

                check_record['done'].append(done)
                if done == 0:
                # critic
                    next_value = self.critic(state.unsqueeze(dim=0).to(self.device))
                    target_value = reward.to(self.device) + 0.9 * next_value
                else:
                    target_value = reward.to(self.device)

                self.actor.optimizer.zero_grad()
                advantage = target_value - value
                actor_loss = -log_prob * advantage.detach()
                actor_loss.backward()
                self.actor.optimizer.step()

                self.critic.optimizer.zero_grad()
                critic_loss = self.loss_function(value, target_value)
                critic_loss.backward()
                self.critic.optimizer.step()

                self.record['pi_loss'].append(actor_loss.item())
                self.record['v_loss'].append(critic_loss.item())
                loss_record['pi_loss'].append(actor_loss.item())
                loss_record['v_loss'].append(critic_loss.item())

                action_transition.append(action)
                             
                if done == 1:
                    for action in action_transition:
                        self.env.graph.edges[action].slot_num -= 1
                    print(check_record)
                    check_record = {'flow_info': [],'action': [],'done': []}
                    break
                elif done == -1:
                    print(check_record)
                    check_record = {'flow_info': [],'action': [],'done': []}
                    break
                elif done == 0:
                    continue

            self.env.renew()

        reward = self.env.total_reward / len(self.data.flow_info)
        success_rate = self.env.success_num / len(self.data.flow_info)
        self.record['reward'].append(reward)
        self.record['rate'].append(success_rate)
        mean_pi_loss = sum(loss_record['pi_loss']) / len(loss_record['pi_loss'])
        mean_v_loss = sum(loss_record['v_loss']) / len(loss_record['v_loss'])

        print(datetime.datetime.now().strftime('\n[%m-%d %H:%M:%S]'),
              'Reward: {:+.02f} |'.format(reward),
              'Rate: {:.02f}'.format(success_rate))
        print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
              'Pi Loss: {:+.04f} |'.format(mean_pi_loss),
              'V Loss: {:.04f}'.format(mean_v_loss))

    @staticmethod
    def select_action(actions, mask):
        probs = F.softmax(actions, dim=0)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return mask[action.item()], log_prob

    def train(self):
        for episode in range(3000):
            episode += 1
            print(datetime.datetime.now().strftime('[%m-%d %H:%M:%S]'),
                  'Episode: {:04d}'.format(episode))

            self.train_one_episode()

            # self.save_record(episode)
            print('#' * 60)

    def save_record(self, episode):
        if episode % 100 == 0:
            if not os.path.exists('record'):
                os.makedirs('record')
            for key in self.record.keys():
                np.save(f'record/{key}_{episode}.npy', self.record[key])


def main():
    # if os.path.exists('record'):
    #     shutil.rmtree('record')
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()