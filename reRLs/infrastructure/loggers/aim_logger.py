import os
import os.path as osp
import numpy as np
import aim

from enum import Enum
from typing import Dict
from moviepy import editor as mpy # this line for 'moviepy' correctly run
from matplotlib import pyplot as plt
from reRLs import user_config as conf
from reRLs.infrastructure.loggers.base_logger import Logger, mkdir_p
from reRLs.infrastructure.utils import utils

def safe_dict(dic):
    def default(o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        else: return o

    for key, val in dic.items():
        dic[key] = default(val)

    return dic

class AimLogger(Logger):
    def __init__(self):
        super().__init__()
        self._aim_run = None

    def set_snapshot_dir(self, dir_name):
        ''' set snapshot dir and init aim run '''
        self._snapshot_dir = dir_name

        exp_name = dir_name.split("/")[-1]
        exp_name_no_timestamp = '_'.join(exp_name.split("_")[-3:-1])
        seed_exp_id = exp_name.split("_")[-1]
        aim_dir = osp.dirname(osp.dirname(osp.dirname(dir_name)))
        self._aim_run = aim.Run(
            repo=aim_dir,
            experiment=exp_name
        )
        self._aim_run.name = exp_name_no_timestamp
        self._aim_run.description = seed_exp_id
        # Add description and change Run name
        print('########################')
        print('aim outputs to ', aim_dir)
        print('########################')

    def log_variant(self, file_name, variant_data):
        # super(AimLogger, self).log_variant(file_name, variant_data)
        super().log_variant(file_name, variant_data)
        assert isinstance(file_name, str), 'file_name must be std'
        assert isinstance(variant_data, dict), 'file_name must be dict'
        file_name = osp.splitext(file_name)[-2] # Remove the extension name which Aim does not support.
        self._aim_run[file_name] = safe_dict(variant_data)

    def log_scalar(self, scalar, name, step_, context=None):
        self._aim_run.track(value=scalar, name=name, step=step_, context=context)

    def log_image(self, image, name, step_, context=None):
        assert(len(image.shape) == 3)  # [C, H, W]
        self._aim_run.track(value=aim.Image(image), name=name, step=step_, context=context)

    def log_video(self, n_video_frames, name, step_, fps=10, context=None):
        assert len(n_video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        aim_images = []
        for idx, video_frames in enumerate(n_video_frames):
            caption = f'{name}_{step_}_{idx}.gif'
            video_log_path = osp.join(self._video_log_dir, caption)
            utils.write_gif(video_log_path, video_frames, fps=fps)
            aim_image = aim.Image(image=video_log_path, caption=caption, format='gif')
            aim_images.append(aim_image)
        self._aim_run.track(value=aim_images, name=name, step=step_, context=context)

    def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):

        # reshape the rollouts
        # videos = [np.transpose(p['image_obs'], [0, 1, 2, 3]) for p in paths]
        videos = [p['image_obs'] for p in paths]

        # max rollout length
        max_videos_to_save = np.min([max_videos_to_save, len(videos)])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0]>max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all be same length
        for i in range(max_videos_to_save):
            if videos[i].shape[0]<max_length:
                padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        # log videos to local dir and aim logger
        videos = np.stack(videos[:max_videos_to_save], 0)
        self._video_log_dir = osp.join(self._snapshot_dir, 'video')
        mkdir_p(self._video_log_dir)
        self.log_video(videos, video_title, step, fps=fps)

    def log_figure(self, figure, name, step, context=None, dpi=300):
        """figure: matplotlib.pyplot figure handle"""
        # aim default matplotlib track is ugly !!
        # self._aim_run.track(value=aim.Figure(figure), name=name, step=step, context=context)
        self._figure_log_dir = osp.join(self._snapshot_dir, 'figure')
        mkdir_p(self._figure_log_dir)
        caption = f'{name}_{step}.png'
        figure_log_path = osp.join(self._figure_log_dir, caption)
        plt.savefig(figure_log_path, dpi=dpi)
        plt.close(figure)
        aim_image = aim.Image(image=figure_log_path, caption=caption, format='png')
        self._aim_run.track(value=aim_image, name=name, step=step, context=context)

    def log_distribution(self, param, name, step, context=None):
        dist = aim.Distribution(distribution=param)
        self._aim_run.track(dist, name=name, step=step, context=context)

    def dump_tabular(self, *args, **kwargs):

        tabular_dict = dict(self._tabular)

        itr_mode = False
        if 'Itr' in tabular_dict.keys(): 
            itr = tabular_dict.pop('Itr')
            itr_mode = True

        step_mode = False
        if 'TotalEnvInteracts' in tabular_dict.keys():
            step_cnt= int(tabular_dict.pop('TotalEnvInteracts'))
            step_mode = True

        for key, value in tabular_dict.items():
            if itr_mode:
                self.log_scalar(value, key, itr, context={'mode': 'Itr'})
            if step_mode:
                self.log_scalar(value, key, step_cnt, context={'mode': 'Step'})

        super().dump_tabular(*args, **kwargs)

    def save_itr_params(self, itr, params):
        super().save_itr_params(itr, params)
        for name, param in params.items():
            self.log_distribution(param, name, itr, context={'mode': 'Itr'})

    def close(self):
        self._aim_run.close()
