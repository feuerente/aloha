import numpy as np
import collections


class FakeEnv:
    TRAJECTORY_PATH = './data/furniture_trajectories_small/traj_0/'
    def __init__(self):
        self.trajectory_path = 'aloha/aloha_scripts/data/furniture_trajectories_small/traj_0/'
        self.qpos = np.load(self.trajectory_path+'qpos.npz')['arr_0']
        self.qvel = np.load(self.trajectory_path+'qvel.npz')['arr_0']
        self.effort = np.load(self.trajectory_path+'effort.npz')['arr_0']
        self.cam_low = np.moveaxis(np.load(self.trajectory_path+'cam_low.npz')['arr_0'],1,-1)
        self.index = 0

    def reset(self,fake=True):
        obs = collections.OrderedDict()
        obs['qpos'] = self.qpos[self.index,...]
        obs['qvel'] = self.qvel[self.index,...]
        obs['effort'] = self.effort[self.index,...]
        obs['cam_low'] = self.cam_low[self.index,...]
        self.index = self.index + 1
        return obs
    
    def step(self):
        return self.reset_env()
    
def make_real_env(init_node, setup_robots=True):
    env = FakeEnv()
    return env
    
