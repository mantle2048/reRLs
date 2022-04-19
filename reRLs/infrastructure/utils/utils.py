import os
import time
import copy
import PIL
import numpy as np
from pyvirtualdisplay import Display
from moviepy.editor import ImageSequenceClip

############################################
############################################
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['obs']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states

def perform_actions(env, actions):
    obs = env.reset()
    obss, image_obss, acts, rews, next_obss, dones = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def mean_squared_error(a, b):
    return np.mean((a-b)**2)

############################################
############################################

def mp_sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    '''mp sample version do not support render'''
    num_envs = env.num_envs
    obs = env.reset()

    obss, acts, rews, next_obss, terminals, image_obss = [], [], [], [], [], []
    steps = 0
    while True:

        obss.append(obs)
        act = policy.get_action(obs)

        acts.append(act)

        next_obs, rew, done, _ = env.step(act)

        rews.append(rew)
        next_obss.append(next_obs)
        obs = next_obs

        steps += 1

        if steps >= max_path_length:
            rollout_done = [True] * num_envs
        else:
            rollout_done = done
        terminals.append(rollout_done)


        if steps >= max_path_length:
            break
    obss = np.stack(obss, axis=1)
    acts = np.stack(acts, axis=1)
    rews = np.stack(rews, axis=1)
    next_obss = np.stack(next_obss, axis=1)
    terminals = np.stack(terminals, axis=1)
    import ipdb; ipdb.set_trace()
    mp_path = [Path(obs, [], act, rew, next_obs, terminal)  \
                for obs, act, rew, next_obs, terminal in  \
                zip(obss, acts, rews, next_obss, terminals)]

    return mp_path

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    obs = env.reset()

    obss, acts, rews, next_obss, terminals, image_obss = [], [], [], [], [], []
    steps = 0
    while True:
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obss.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                else:
                    image_obss.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)

        obss.append(obs)
        act = policy.get_action(obs)
        if len(act.shape) > 1:
            act = act[0]
        acts.append(act)

        next_obs, rew, done, _ = env.step(act)

        rews.append(rew)
        next_obss.append(next_obs)
        obs = next_obs

        steps += 1

        rollout_done = done or steps >= max_path_length
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obss, image_obss, acts, rews, next_obss, terminals)

def mp_sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    paths = []
    timesteps_this_batch = 0
    while timesteps_this_batch < min_timesteps_per_batch:
        # cur_path_length = min(max_path_length, min_timesteps_per_batch - timesteps_this_batch)
        mp_path = mp_sample_trajectory(env, policy, max_path_length, render, render_mode)
        for path in mp_path:
            timesteps_this_batch += get_pathlength(path)
            paths.append(path)
            if timesteps_this_batch >= min_timesteps_per_batch:
                break

    return paths, timesteps_this_batch

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    paths = []
    timesteps_this_batch = 0
    while timesteps_this_batch < min_timesteps_per_batch:
        # cur_path_length = min(max_path_length, min_timesteps_per_batch - timesteps_this_batch)
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        timesteps_this_batch += get_pathlength(path)
        paths.append(path)

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    paths = [ sample_trajectory(env, policy, max_path_length, render, render_mode) for _ in range(ntraj) ]
    return paths

############################################
############################################

def Path(obss, image_obss, acts, rews, next_obss, dones):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obss != []:
        image_obss = np.stack(image_obss, axis=0)

    return {"obs" : np.array(obss, dtype=np.float32),
            "image_obs" : np.array(image_obss, dtype=np.uint8),
            "act" : np.array(acts, dtype=np.float32),
            "rew" : np.array(rews, dtype=np.float32),
            "next_obs": np.array(next_obss, dtype=np.float32),
            "done": np.array(dones, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    obss = np.concatenate([path["obs"] for path in paths])
    acts = np.concatenate([path["act"] for path in paths])
    next_obss = np.concatenate([path["next_obs"] for path in paths])
    dones = np.concatenate([path["done"] for path in paths])
    concated_rews = np.concatenate([path["rew"] for path in paths])
    unconcated_rews = [path["rew"] for path in paths]
    return obss, acts, concated_rews, unconcated_rews, next_obss, dones


############################################
############################################

def get_pathlength(path):
    return len(path["rew"])

def standardize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def de_standardize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data

############################################
############################################

def write_gif(filename, array, fps=10, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64, 3)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # Low quality but quick gif save by using PIL
    imgs = [PIL.Image.fromarray(img) for img in array]
    imgs[0].save(filename, save_all=True, append_images=imgs[1:], duration=1000//fps, loop=0)
    return filename

    # High quality but slow gif save by using moviepy
    # clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    # clip.write_gif(filename, fps=fps)
    # return clip
