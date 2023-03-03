import numpy as np
import matplotlib.pyplot as plt


episode = 3000


def move_average(data, window_size):
    return np.convolve(data, np.ones(int(window_size)) / float(window_size), 'same')


def draw(data_type):
    window_size = 3000 if data_type == 'pi_loss' or data_type == 'v_loss' else 10
    data = np.load(f'record/{data_type}_{episode}.npy', allow_pickle=True)
    y = move_average(data, window_size)
    # y = data.mean()
    # y_mean = y.mean()
    y_std = np.std(data, axis = 0)
    # print(y_std)
    # print(len(y))
    x = np.arange(len(y))

    plt.plot(x, y, 'r')
    plt.xlabel('episode')
    plt.ylabel(data_type)
    
    # plt.fill_between(x, y - y_std, y + y_std, color='lightcoral', alpha=0.2)
    
    length = len(y) if data_type == 'pi_loss' or data_type == 'v_loss' else episode
    plt.xlim(int(window_size / 2), int(length - (window_size / 2)))
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    draw('reward')
    draw('rate')
    draw('pi_loss')
    draw('v_loss')