# +
# %matplotlib notebook
# %matplotlib inline


# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

# %load_ext autoreload
# %autoreload 2

from typing import List, Dict

import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


data_dir = os.path.join(os.getcwd(), 'run_logs')
img_dir = os.path.join(os.getcwd(), 'image')


# +
def get_event_data(event_dir: str) -> pd.DataFrame:
    from collections import defaultdict
    exp_name = event_dir.split('/')[-3]
    run_name = event_dir.split('/')[-2]
    data_dict = defaultdict(list)

    Iteration = 0
    for event in tf.train.summary_iterator(event_dir):
        for value in event.summary.value:
            if value.tag is not None:
                data_dict[value.tag].append(value.simple_value)

    step_length = None
    for key, values in data_dict.items():
        if key == 'Train_EnvstepsSoFar':
            step_length = len(values)

    assert step_length is not None, 'Please specify your steps key value'

    for key, values in data_dict.items():
        if len(values) < step_length:
            gap =  step_length - len(values)
            data_dict[key] = [0] * gap + values


    iteration = np.arange(1, step_length + 1)
    data_dict['Iteration'] = iteration


    data = pd.DataFrame(data_dict)
    data['Condition1'] = exp_name
    data['Condition2'] = run_name
    return data

# event_dir = '/home/yan/code/CS285_2020Fall_Homework/hw5/run_logs/part2_sub1/hw5_expl_q2_med_dqn/hw5_expl_q2_med_dqn_PointmassMedium-v0_06-01-2022_14-34-49/events.out.tfevents.1641450889.yan-PC'
# get_event_data(event_dir)
# -

def get_exp_data(events_dir: str) -> List[pd.DataFrame]:
    exp_data = []
    for event_dir in events_dir:
        exp_data.append(get_event_data(event_dir))
    return exp_data


def plot_data(data: pd.DataFrame, ax=None, xaxis='Iteration', value='EvalAverageReturn', condition='Condition2', smooth=1):
    if smooth > 1:
        if isinstance(data, list):
            for datam in data:
                datam[value]=uniform_filter1d(datam[value], size=smooth)
        else:
            data[value]=uniform_filter1d(data[value], size=smooth)
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    env_name = '_'.join(data['Condition1'][0].split('_')[0:-1])
    sns.set(style='whitegrid', palette='tab10', font_scale=1.5)
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', ax=ax, linewidth=3.0)
    leg = ax.legend(loc='best') #.set_draggable(True)
    for line in leg.get_lines():
        line.set_linewidth(3.0)
    ax.set_title(env_name)
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


def sort_func(x):
    suffix = x.split('/')[-3].split('_')[-1]
    try:
        suffix=float(suffix)
    except ValueError:
        print('exp_suffix is not a str')
    return suffix


# +
def read_exp_data_and_plot(log_dir: str, exp_names: List, plot_config: Dict={}):
    exp_num = len(exp_names)
    fig, axs = plt.subplots(exp_num, 1, figsize=(8, exp_num * 6))
    if not isinstance(axs, np.ndarray): axs = [axs]
    for ax, exp_name in zip(axs, exp_names):
        exp_dir = os.path.join(log_dir, f'*{exp_name}*')
        events_dir = glob.glob(os.path.join(exp_dir, '*', '*events*'))
        events_dir = sorted(events_dir, key=sort_func, reverse=False)
        exp_data = get_exp_data(events_dir)
        plot_data(data=exp_data, ax=ax, **plot_config)
    plt.tight_layout(pad=0.5)


if __name__ == '__main__':
    pass


