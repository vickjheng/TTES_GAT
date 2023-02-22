import numpy as np
import matplotlib.pyplot as plt


episode = 2000


def move_average(data, window_size):
    return np.convolve(data, np.ones(int(window_size)) / float(window_size), 'same')


def draw(data_type):
    window_size = 200 if data_type == 'pi_loss' or data_type == 'v_loss' else 10
    y = move_average(np.load(f'record/{data_type}_{episode}.npy', allow_pickle=True), window_size)
    print(len(y))
    x = np.arange(len(y))
    plt.plot(x, y, 'r')
    plt.xlabel('episode')
    plt.ylabel(data_type)
    length = len(y) if data_type == 'pi_loss' or data_type == 'v_loss' else episode
    plt.xlim(int(window_size / 2), int(length - (window_size / 2)))
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    draw('reward')
    draw('rate')
    draw('pi_loss')
    draw('v_loss')