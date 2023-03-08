import numpy as np
import matplotlib.pyplot as plt


episode = 10000
window_size = 500


def move_average(data):
    return np.convolve(data, np.ones(int(window_size)) / float(window_size), 'same')


def draw(data):
    datas = np.load(f'train_record_gua/{data}_{episode}.npy', allow_pickle=True)
    y = move_average(datas)
    x = np.arange(len(y))
    # if data == 'loss':
    #     y_std = np.std(datas, axis = 0)
    #     plt.fill_between(x, y - y_std, y + y_std, color='lightcoral', alpha=0.2)
    plt.plot(x, y, 'r')
    plt.xlim(int(window_size / 2), int(episode - (window_size / 2)))
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    draw('loss')
    # draw('reward')
    # draw('success_rate')