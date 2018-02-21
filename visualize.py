# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import json
import os

import numpy as np
from scipy.signal import medfilt


def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy

def load_csv(log_folder):
    datas = []
    monitor_files = glob.glob(os.path.join(log_folder, '*.monitor.csv'))

    for input_file in monitor_files:
        with open(input_file, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    return result, timesteps

def load_data(log_folder, smooth, bin_size):
    result, timesteps = load_csv(log_folder)

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


def moving_average(values, window):
    """
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def episode_plot(viz, win, folder, game, name, window=5, title=""):
    """
    Create/Update a vizdom plot of reward per episode
    :param viz: (visdom object)
    :param win: (str) Window name, it is the unique id of each plot
    :param folder: (str) Log folder, where the monitor.csv is stored
    :param game: (str) Name of the environment
    :param name: (str) Algo name
    :param window: (int) Smoothing window
    :param title: (str) additional info to display in the plot title
    :return: (str)
    """
    result, _ = load_csv(folder)

    if len(result) == 0:
        return win

    y = np.array(result)[:, 1]
    x = np.arange(len(y))

    if y.shape[0] < window:
        return win

    y = moving_average(y, window)

    if len(y) == 0:
        return win

    # Truncate x
    x = x[len(x) - len(y):]
    opts = {
        "title": "{}\n{}".format(game, title),
        "xlabel": "Number of Episodes",
        "ylabel": "Rewards",
        "legend": [name]
    }
    return viz.line(y, x, win=win, opts=opts)


def visdom_plot(viz, win, folder, game, name, bin_size=100, smooth=1, title=""):
    """
    Create/Update a vizdom plot of reward per timesteps
    :param viz: (visdom object)
    :param win: (str) Window name, it is the unique id of each plot
    :param folder: (str) Log folder, where the monitor.csv is stored
    :param game: (str) Name of the environment
    :param name: (str) Algo name
    :param bin_size: (int)
    :param smooth: (int) Smoothing method (0 for no smoothing)
    :param title: (str) additional info to display in the plot title
    :return: (str)
    """
    tx, ty = load_data(folder, smooth, bin_size)
    if tx is None or ty is None:
        return win

    if len(tx) * len(ty) == 0:
        return win

    tx, ty = np.array(tx), np.array(ty)

    opts = {
        "title": "{}\n{}".format(game, title),
        "xlabel": "Number of Timesteps",
        "ylabel": "Rewards",
        "legend": [name]
    }
    return viz.line(ty, tx, win=win, opts=opts)


if __name__ == "__main__":
    from visdom import Visdom
    viz = Visdom()
    visdom_plot(viz, None, '/tmp/gym/', 'BreakOut', 'a2c', bin_size=100, smooth=1)
