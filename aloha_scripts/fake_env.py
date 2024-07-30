import collections

import dm_env
import numpy as np


class FakeEnv:
    def __init__(
            self,
            left_arm_only: bool = False
    ):
        self.left_arm_only = left_arm_only

        self.trajectory_path = '/home/ralf/projects/alr_prak/trajectory-diffusion-prak.git/_data/traj_2024-07-12_single-leg/traj_0'
        # TODO Check which files are in the trajectory dir
        self.qpos = np.load(self.trajectory_path + '/qpos.npz')['arr_0']
        self.qvel = np.load(self.trajectory_path + '/qvel.npz')['arr_0']
        self.effort = np.load(self.trajectory_path + '/effort.npz')['arr_0']
        self.parts_poses = np.load(self.trajectory_path + '/parts_poses.npz')['arr_0']
        # self.cam_low = np.moveaxis(np.load(self.trajectory_path + '/cam_low.npz')['arr_0'], 1, -1)
        self.index = 0

    def reset(self, fake=False):
        self.index = 0
        return self.next_step()

    def step(self, action, move_time_arm, move_time_gripper):
        return self.next_step()

    def next_step(self):
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def get_reset_action(self):
        start_arm_pose = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
        puppet_gripper_joint_close = -0.6213
        reset_action_single_arm = start_arm_pose[:6] + [puppet_gripper_joint_close]
        if self.left_arm_only:
            return reset_action_single_arm
        return reset_action_single_arm + reset_action_single_arm

    def get_observation(self):
        cutoff = 7 if self.left_arm_only else 14
        obs = collections.OrderedDict()
        obs['qpos'] = self.qpos[self.index][:cutoff]
        obs['qvel'] = self.qvel[self.index][:cutoff]
        obs['effort'] = self.effort[self.index][:cutoff]
        obs['images'] = {}
        # obs['images']['cam_low'] = self.cam_low[self.index]
        obs['parts_poses'] = self.parts_poses[self.index]
        self.index = self.index + 1
        return obs

    def get_reward(self):
        return 0


def make_real_env(init_node, furniture="square_table", setup_robots=True, left_arm_only=False):
    env = FakeEnv(left_arm_only)
    return env
