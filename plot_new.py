import numpy as np
import matplotlib.pyplot as plt
from param import args

episode = 3000
window_size = episode / 20


def move_avg(data):
    window = np.ones(int(window_size)) / float(window_size)

    return np.convolve(data, window, 'same')


def draw_env_record():
    env_record = {'success_rate': [], 'usage': [], 'reward': []}
    for key in env_record.keys():
        env_record[key].extend(move_avg(np.load(f'record/env/{key}_{episode}.npy', allow_pickle=True)))
    x = np.arange(len(env_record['success_rate']))
    fig, plot = plt.subplots(len(env_record))
    for idx, key in zip(range(len(env_record)), env_record.keys()):
        plot[idx].plot(x, env_record[key], 'r')
        plot[idx].set_xlim(int(window_size / 2), int(episode - (window_size / 2)))
        plot[idx].grid(True)
        plot[idx].set_title(f'{key}')
    fig.tight_layout()
    
    y = move_avg(np.load(f'record/loss_{episode}.npy'))
    x = np.arange(len(y))
    plt.figure()
    plt.plot(x, y,color='r',label = 'loss')
    
    # lr = np.load(f'record/lr_{episode}.npy')
    # x_lr = np.arange(args.exploration_end_episode,args.exploration_end_episode+len(lr)) 
    # plt.plot(x_lr, lr,color='green',label = 'lr')

    # epsilon = np.load(f'record/epsilon_{episode}.npy')
    # x_esp = np.arange(args.exploration_end_episode,args.exploration_end_episode+len(epsilon)) 
    # plt.plot(x_esp,epsilon,color ='b',label = 'epsilon')
    
    plt.xlim(int(window_size / 2), int(episode - (window_size / 2)))
    plt.title('Loss')
    #plt.axvline(args.OBSERVE, linestyle= '--',label='end_exploration')
    plt.grid(True)
    plt.legend(loc='best') 
    plt.show()


if __name__ == '__main__':
    draw_env_record()
    # draw_training_record()
