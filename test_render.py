import pybullet_envs
import gym
import dmc2gym
from pyvirtualdisplay import Display

def test_bullet_render():

    env = gym.make("MinitaurBulletEnv-v0", disable_env_checker=True)
    obs = env.reset()

    img = env.render(mode='rgb_array', height=500)
    import ipdb; ipdb.set_trace()


    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward)

        if done:
            obs = env.reset()
        env.close()

def test_dmc_render():
    env = dmc2gym.make(domain_name="cheetah", task_name="run")
    import ipdb; ipdb.set_trace()
    obs = env.reset()

    img = env.render(mode='rgb_array')

    return img

if __name__ == '__main__':
    virtual_disp = Display(visible=False, size=(1400,900))
    virtual_disp.start()
    img = test_dmc_render()
    print(img.shape)
