import math
from collections import defaultdict, deque
from functools import partial
from typing import List

import numpy as np
import modern_robotics as mr
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from trajectory_diffusion.datasets.scalers import standardize, normalize, denormalize, destandardize
from trajectory_diffusion.utils.helper import deque_to_array

# Use fake_env for testing
from real_env import make_real_env
# from fake_env import make_real_env


# TODO Record trajectory and timing

class RobotTester:
    def __init__(self, cfg):
        hydra_config = cfg["hydra_config"]
        dataset_config = hydra_config["dataset_config"]

        self.keys = dataset_config["keys"]
        self.max_action_steps = cfg["max_action_steps"]
        self.left_arm_only = cfg["left_arm_only"]
        self.t_obs = hydra_config["t_obs"]
        self.t_act = hydra_config["t_act"]
        self.move_time_arm = cfg["move_time_arm"]
        self.move_time_gripper = cfg["move_time_gripper"]
        self.parts_poses_euler = "parts_poses_euler" in hydra_config["dataset_config"] and \
                                 hydra_config["dataset_config"]["parts_poses_euler"]
        self.relative_action_values = "relative_action_values" in hydra_config["agent_config"][
            "process_batch_config"] and hydra_config["agent_config"]["process_batch_config"]["relative_action_values"]

        self.env = make_real_env(init_node=True, furniture="table_leg", setup_robots=True,
                                 left_arm_only=self.left_arm_only)

        self.observation_buffer = defaultdict(partial(deque, maxlen=self.t_obs))

        # SCALING AND NORMALIZATION
        self.normalize_keys: List[str] = dataset_config["normalize_keys"] if isinstance(
            dataset_config["normalize_keys"], list) else []
        self.standardize_keys: List[str] = dataset_config["standardize_keys"] if isinstance(
            dataset_config["standardize_keys"], list) else []
        self.normalize_symmetrically = dataset_config["normalize_symmetrically"]

        self.scaler_values = hydra_config["workspace_config"]["env_config"]["scaler_config"]["scaler_values"]

        for key in self.normalize_keys:
            assert key in self.scaler_values, f"Key {key} not found in scaler values."
            for metric in ["min", "max"]:
                assert metric in self.scaler_values[key] and self.scaler_values[key][
                    metric] is not None, f"Key {key} does not have {metric} in scaler values."
                self.scaler_values[key][metric] = np.array(self.scaler_values[key][metric], dtype=np.float32)

        for key in self.standardize_keys:
            assert key in self.scaler_values, f"Key {key} not found in scaler values."
            for metric in ["mean", "std"]:
                assert metric in self.scaler_values[key] and self.scaler_values[key][
                    metric] is not None, f"Key {key} does not have {metric} in scaler values."
                self.scaler_values[key][metric] = np.array(self.scaler_values[key][metric], dtype=np.float32)

    def test_agent(self, agent):
        """Run agent on robot"""
        ts = self.env.reset(fake=False)
        last_desired_absolute_action = np.array(self.env.get_reset_action(), dtype=np.float32)[None]
        # Preload observations
        for i in range(self.t_obs):
            self.update_observation_buffer(ts.observation)

        # Set the agent's weights to the EMA weights, without storing and loading them in very call to predict()
        agent.use_ema_weights()

        max_action_sequences = math.ceil(self.max_action_steps / self.t_act)
        action_step_counter = 0

        progress_bar = tqdm(total=self.max_action_steps, desc="Testing Agent", unit="action")
        for action_sequence in range(max_action_sequences):
            observation = deque_to_array(self.observation_buffer)

            # Create torch tensors from numpy arrays
            for key, val in observation.items():
                observation[key] = torch.from_numpy(val)

            observation, extra_inputs = agent.process_batch.process_env_observation(observation)

            # Move observations and extra inputs to device
            for key, val in observation.items():
                observation[key] = val.to(agent.device)
            for key, val in extra_inputs.items():
                if isinstance(val, torch.Tensor):
                    extra_inputs[key] = val.to(agent.device)

            # Predict the next action sequence
            actions = agent.predict(observation, extra_inputs)

            # Remove batch dimension and move to cpu and get numpy array
            actions = actions.squeeze(0).cpu().numpy()

            actions = self.denormalize_destandardize_actions(actions)

            if self.relative_action_values:
                actions += last_desired_absolute_action
                last_desired_absolute_action += np.sum(actions, axis=0)

            for action in actions[: self.t_act]:
                if action_step_counter >= self.max_action_steps:
                    break

                # current_pos = self.env.puppet_bot_left.arm.get_ee_pose()
                # destination = mr.FKinSpace(self.env.puppet_bot_left.arm.robot_des.M, self.env.puppet_bot_left.arm.robot_des.Slist, action[0:6])

                # Check height (Warning: EEF height not tip of the gripper)
                # if (height := destination[2, 3]) < 0.05:
                #     print(f"Warning: Height ({height}) will be too low")
                #     while True:
                #         if input("Press enter to continue") == "":
                #             break

                ts = self.env.step(action, move_time_arm=self.move_time_arm, move_time_gripper=self.move_time_gripper)
                self.update_observation_buffer(ts.observation)

                action_step_counter += 1
                progress_bar.update()

        # Restore the original weights
        agent.restore_model_weights()

        # TODO
        # return result_dict

    def denormalize_destandardize_actions(self, action: np.ndarray) -> np.ndarray:
        if (key := "action") in self.normalize_keys:
            action = denormalize(action, self.scaler_values[key], symmetric=self.normalize_symmetrically)
        if (key := "action") in self.standardize_keys:
            action = destandardize(action, self.scaler_values[key])
        return action

    def update_observation_buffer(self, observation):
        for key, value in observation.items():
            if key not in self.keys:
                continue

            value = np.array(value, dtype=np.float32)

            if key == "parts_poses" and self.parts_poses_euler:
                value = self.parts_poses_to_euler(value)

            # TODO adjust images

            if key in self.normalize_keys:
                value = normalize(value, self.scaler_values[key], self.normalize_symmetrically)
            if key in self.standardize_keys:
                value = standardize(value, self.scaler_values[key])

            self.observation_buffer[key].append(value)

    def parts_poses_to_euler(self, parts_poses):
        out_parts_poses = []
        for part_pose in parts_poses.reshape(-1, 7):
            quaternion = part_pose[3:7]
            euler_angle = R.from_quat(quaternion).as_euler('xyz', degrees=False)
            out_parts_poses.append(np.concatenate((part_pose[:3], euler_angle)))
        return np.hstack(out_parts_poses, dtype=np.float32)

# def adjust_images(observation):
#     # takes all images from the observation and transforms it to the correct shape, i.e. (#images, channels, ...)
#     # for key in observations: #use key images on real stuff ['images']
#     #     observations[key] = np.moveaxis(observations[key],-1,1) #dont forgetkey images
#
#     # for cam_name in camera_names:
#     #     observation['images'][cam_name] = np.moveaxis(observation['images'][cam_name],-1,0)[...,:CROP_SIZES[0],:CROP_SIZES[1]]
#     # TODO normalize pixels to [0,1] by dividing by 255
#     for key, value in observation.items():
#         if key == 'images':
#             # for cam_name in IMAGE_KEYS:
#             #     observation[key][cam_name] = torch.tensor(value[cam_name])[...]
#             continue
#         observation[key] = torch.tensor(value, dtype=torch.float32)[...]
#     # for cam_name in IMAGE_KEYS:
#     #     observation['images'][cam_name] = observation['images'][cam_name].type(torch.FloatTensor)
#     return observation


# def get_auto_index(dataset_dir, dataset_name_prefix='', data_suffix='hdf5'):
#     max_idx = 1000
#     if not os.path.isdir(dataset_dir):
#         os.makedirs(dataset_dir)
#     for i in range(max_idx + 1):
#         if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
#             return i
#     raise Exception(f"Error getting auto index, or more than {max_idx} episodes")
#
#


# # save the data
# data_dict = {
#     '/observations/qpos': [],
#     '/observations/qvel': [],
#     '/observations/effort': [],
#     '/observations/parts_poses': [],
#     '/action': [],

# }
# for cam_name in CAMERA_NAMES:
#     data_dict[f'/observations/images/{cam_name}'] = []

# # len(action): max_timesteps, len(time_steps): max_timesteps + 1
# while actions:
#     action = actions.pop(0)
#     ts = timesteps.pop(0)
#     data_dict['/observations/qpos'].append(ts['qpos'])
#     data_dict['/observations/qvel'].append(ts['qvel'])
#     data_dict['/observations/effort'].append(ts['effort'])
#     data_dict['/observations/parts_poses'].append(ts['parts_poses'])
#     data_dict['/action'].append(action)
#     for cam_name in CAMERA_NAMES:
#         data_dict[f'/observations/images/{cam_name}'].append(ts['images'][cam_name])

# # HDF5
# t0 = time.time()
# index = get_auto_index(DATASET_DIR)
# dataset_path = DATASET_DIR + f'episode_{index}'
# with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
#     root.attrs['sim'] = False
#     obs = root.create_group('observations')
#     image = obs.create_group('images')
#     for cam_name in CAMERA_NAMES:
#         _ = image.create_dataset(cam_name, (MAX_TIMESTEPS, 480, 640, 3), dtype='uint8',
#                                  chunks=(1, 480, 640, 3), )
#     number_joints = 7 if HALVED_POLICY else 14
#     _ = obs.create_dataset('qpos', (MAX_TIMESTEPS, number_joints))
#     _ = obs.create_dataset('qvel', (MAX_TIMESTEPS, number_joints))
#     _ = obs.create_dataset('effort', (MAX_TIMESTEPS, number_joints))
#     _ = obs.create_dataset('parts_poses', (MAX_TIMESTEPS, 7))
#     _ = root.create_dataset('action', (MAX_TIMESTEPS, number_joints))

#     for name, array in data_dict.items():
#         root[name][...] = array
# print(f'Saving: {time.time() - t0:.1f} secs')
