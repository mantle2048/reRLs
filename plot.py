# +
from typing import List, Dict

from typing import List
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path
from termcolor import colored

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10
sns.set(style='whitegrid', palette='tab10', font_scale=1.5)


# %matplotlib notebook
# %reload_ext autoreload
# %autoreload 2


data_dir = osp.join(os.getcwd(), 'benchmark')
event_dir = osp.join(data_dir, 'LunarLander-v2')
print(event_dir)

# -

COLORS = (
    [
        # deepmind style
        '#0072B2',
        '#009E73',
        '#D55E00',
        '#CC79A7',
        # '#F0E442',
        '#d73027',  # RED
        # built-in color
        'blue',
        'red',
        'pink',
        'cyan',
        'magenta',
        'yellow',
        'black',
        'purple',
        'brown',
        'orange',
        'teal',
        'lightblue',
        'lime',
        'lavender',
        'turquoise',
        'darkgreen',
        'tan',
        'salmon',
        'gold',
        'darkred',
        'darkblue',
        'green',
        # personal color
        '#313695',  # DARK BLUE
        '#74add1',  # LIGHT BLUE
        '#f46d43',  # ORANGE
        '#4daf4a',  # GREEN
        '#984ea3',  # PURPLE
        '#f781bf',  # PINK
        '#ffc832',  # YELLOW
        '#000000',  # BLACK
    ]
)


# +
def compute_std(data_frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    '''compute and return standard deviation of give the list of data_frame'''
    data_dict = defaultdict(list)
    std_data_dict = {}
    for data_frame in  data_frames:
        for key in data_frame.keys():
            data_dict[key].append(data_frame[key].to_list())
            
    for key in data_dict:
        data_list = data_dict[key]
        pad = len(max(data_list, key=len))
        data_array = np.array([i + [0]*(pad-len(i)) for i in data_list])
        std_data_dict['std:'+key] = np.std(data_array, axis=0)
    
    std_data_frame = pd.DataFrame(std_data_dict)
    return std_data_frame

def compute_mean(data_frames: List[pd.DataFrame], key='TotalEnvInteracts') -> List[pd.DataFrame]:
    value_list = []
    for data_frame in  data_frames:
        value_list.append(data_frame[key].to_list())
            
    value_array = np.array(value_list)
    mean_value = np.mean(value_array, axis=0)
    for data_frame in  data_frames:
        data_frame[key] = mean_value


# -

def get_exp_data(exp_dir, condition=None):
    '''data_dir: ~/{data_dir}/{event_dir}/{exp_name}/{run_name}
    Note that in exp_dir, may be multi runs with
    different random seeds
    example: ~/data/LunarLander-v2/PPO_LunarLander-v2/{PPO_LunarLander-v2_3}
    '''
    progress_dir = osp.join(exp_dir, '*', 'progress.csv')
    all_progress_dirs = glob.glob(progress_dir)
    all_data_frames = []
    for progress_dir in all_progress_dirs:
        run_name = osp.split(osp.dirname(progress_dir))[-1]
        exp_name = osp.split(osp.dirname(osp.dirname(progress_dir)))[-1]
        condition1 = condition or exp_name
        condition2 = run_name
        data_frame = pd.read_csv(progress_dir, sep=',')
        data_frame['Condition1'] = condition1
        data_frame['Condition2'] = condition2
        all_data_frames.append(data_frame)
    
    # calculate mean totalenvinteracts
    compute_mean(all_data_frames, key='TotalEnvInteracts')
    exp_data = pd.concat(all_data_frames, ignore_index=True)
    exp_data.fillna(0)
    return exp_data
exp_data = get_exp_data(osp.join(event_dir, 'PPO_LunarLander-v2'))


def plot_exp_data(
        data: pd.DataFrame, 
        ax=None, 
        xaxis='Itr', 
        value='AverageTrainReward', 
        condition='Condition2', 
        color='blue', 
        smooth=1
    ):
    if smooth > 1:
        if isinstance(data, list):
            for datam in data:
                datam[value]=uniform_filter1d(datam[value], size=smooth)
        else:
            data[value]=uniform_filter1d(data[value], size=smooth)
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', ax=ax, palette=[color], linewidth=3.0)
    leg = ax.legend(loc='best') #.set_draggable(True)
    for line in leg.get_lines():
        line.set_linewidth(3.0)
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


def plot_event_data(
    data_dir,
    event_name,
    xaxis='TotalEnvInteracts',
    value='AverageReward',
    xlabel='Million Steps',
    ylabel='Reward',
    xlim=None,
    ylim=None,
    legend=None,
    title=None,
    count=True,
    smooth=1,
    output_path=None,
    ):
    
    # ===============================================================================
    # get exp_names
    # ===============================================================================
    event_dir = osp.join(data_dir, event_name)
    exp_names = os.listdir(event_dir)
    fig, ax = plt.subplots(1,1)
    
    # ===============================================================================
    # figure setting
    # ===============================================================================
    if xlabel is None:
        xlabel = xaxis 
    if ylabel is None:
        ylabel = value 
        
    if xlim:
        ax.set(xlim=(xlim))
    if ylim:
        ax.set(ylim=(ylim))
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if legend is not None:
        assert len(legend) == len(exp_names), 'Legend number must match exp_name'
    else:
        legend = [None] * len(exp_names)
    
    if title is None:
        title = event_name
        ax.set_title(title)
        
    if output_path is None:
        img_dir = osp.join(data_dir, 'img')
    else:
        img_dir = osp.join(data_dir, 'output_path')
    
    # sort exp_names and legend
    exp_names = sorted(exp_names)
    legend = sorted(legend)
       
    # ===============================================================================
    # read and plot exp data
    # ===============================================================================
    exp_datas = []
    for exp_name, leg in zip(exp_names, legend):
        exp_dir = osp.join(event_dir, exp_name)
        exp_data = get_exp_data(exp_dir, leg)
        exp_datas.append(exp_data)
        
    condition = 'Condition2' if count else 'Condition1'
    
    for idx,(exp_name, exp_data) in enumerate(zip(exp_names, exp_datas)):
        print(exp_name, ': ', exp_data.keys().to_list())
        if value not in exp_data.keys():
            print(colored(f'Fail! {exp_name} doesn\'t have value {value}', 'red'))
            continue
        color = COLORS[idx % len(COLORS)]
        plot_exp_data(data=exp_data, ax=ax, xaxis=xaxis,value=value, condition=condition, color=color, smooth=smooth)
        
    Path(img_dir).mkdir(parents=True, exist_ok=True)
    img_name = osp.join(img_dir, f'{event_name}_{value}.png')
    
    fig.savefig(img_name, dpi=200)
        


def main():
    config = {
        'data_dir': "benchmark",
        'event_name': 'LunarLander-v2',
        'xaxis': 'TotalEnvInteracts',
        'value': 'AverageTrainReward',
        'xlim': [0,300000],
        'ylim': [-200,300],
        'xlabel': None,
        'ylabel': None,
        'title': None,
        'count': False,
        'title': None,
        'smooth': 1,
        'legend': ['REINFORCE','A2C','TRPO', 'PPO'],
        'output_path': None,
    }
    plot_event_data(
        data_dir=config['data_dir'],
        event_name=config['event_name'],
        xaxis=config['xaxis'],
        value=config['value'],
        xlim=config['xlim'],
        ylim=config['ylim'],
        xlabel=config['xlabel'],
        ylabel=config['ylabel'],
        title=config['title'],
        count=config['count'],
        smooth=config['smooth'],
        legend=config['legend'],
        output_path=config['output_path'],
   )


if __name__ == '__main__':
    main()


