import hydra
import torch
import logging
import h5py

from omegaconf import DictConfig, OmegaConf
import numpy as np
import random
import copy

from trajectory_diffusion.utils.setup_helper import setup_agent_and_workspace, parse_wandb_to_hydra_config
from real_env import make_real_env
# from fake_env import make_real_env
from tqdm import tqdm
import modern_robotics as mr
import time
import os
from trajectory_diffusion.datasets.scalers import standardize, normalize, denormalize, destandardize
from trajectory_diffusion.datasets.trajectory_dataset import TrajectoryDataset

from torchvision import transforms
from typing import Dict, Tuple, Optional, List, Union, Any

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("eval", eval)

CONFIG = "test_trained_agent_in_env_furniture"
MAX_TIMESTEPS = 1000
CAMERA_NAMES = []  # 'cam_low','cam_high', 'cam_left_wrist', 'cam_right_wrist'
DATASET_DIR = 'data/task_1/'

MOVE_TIME_ARM = 0 #.1  # in seconds
MOVE_TIME_GRIPPER = 0  # in seconds
HALVED_POLICY = True


@hydra.main(version_base=None, config_path="/home/studentgroup1/trajectory-diffusion-prak/conf", config_name=CONFIG)
def main(cfg: DictConfig) -> None:
    # Load wandb config
    wandb_config = OmegaConf.load(cfg.config)

    # Parse wandb config to hydra config
    hydra_config = parse_wandb_to_hydra_config(wandb_config)

    # Seeds:
    if "seed" in hydra_config:
        torch.manual_seed(hydra_config.seed)
        np.random.seed(hydra_config.seed)
        random.seed(hydra_config.seed)

    # Update config with new values
    if cfg.get("config_to_change") is not None:
        hydra_config = OmegaConf.merge(hydra_config, cfg.to_change)

    # hydra_config['device'] = 'cpu'

    t_obs = hydra_config.get("t_obs")
    t_act = hydra_config.get("t_act")
    relative_action = "relative_action_values" in hydra_config.agent_config.process_batch_config and hydra_config.agent_config.process_batch_config.relative_action_values
    # Setup agent, and workspace
    agent, workspace = setup_agent_and_workspace(hydra_config)
    # setup constant values
    read_config(hydra_config)
    # Load the weights
    agent.load_pretrained(cfg.weights)

    # prepare the real environment
    env = make_real_env(init_node=True, furniture="table_leg", setup_robots=True)
    # Data collection
    ts = env.reset(fake=False)
    timesteps = [ts.observation]
    for i in range(t_obs - 1):
        ts = env.get_observation()
        timesteps.append(ts)
    image_transforms = {}
    image_shapes = {}
    for key in IMAGE_KEYS:
        image_transforms[key], image_shapes[key] = compose_image_transform(timesteps[0]['images'], key, CROP_SIZES, CROP_SIZES, RANDOM_CROP, NORMALIZE_IMAGES)
    for ts in timesteps:
        if HALVED_POLICY:
            for key, value in ts.items():
                if key != 'images' and key != 'parts_poses':
                    ts[key] = value[:value.shape[0] // 2]
                #TODO Images
            #ts = {key: value[:value.shape[0] // 2] for key, value in ts.items()}
        transform_last_image(ts, image_transforms)
        normalize_last_observation(ts)
        standardize_last_observation(ts)
    actions = [torch.zeros(14)]
    actual_dt_history = []
    observations = [adjust_images(state) for state in copy.deepcopy(timesteps)]  # use ts.observation on real_env

    for t in tqdm(range(MAX_TIMESTEPS)):
        t0 = time.time()  #
        last_obs = last_observations(observations[-t_obs:])
        last_obs.pop('effort', None)
        last_obs = {key: value.to(torch.device('cuda')) for key, value in last_obs.items()}
        
        # TODO image_transforms
        action = torch.squeeze(agent.predict(observation=last_obs, extra_inputs=dict()))
        action = action.to(torch.device('cpu'))
        # unnormalize and unstandardize the action if necessary
        if NORMALIZE_ACTION:
            action = denormalize(action, SCALER_VALUES['action'], symmetric=NORMALIZE_SYMMETRICALLY)
        if STANDARDIZE_ACTION:
            action = destandardize(action, SCALER_VALUES['action'])
         #need to make the action shape for 2 robots, simply copy the action for the second robot even tho it wont be used
        last_qpos = last_obs['qpos'].to(torch.device('cpu'))[-1,-1,:]#torch.tensor(env.puppet_bot_left.arm.get_joint_commands())#
        if len(action.size()) == 1:
            action = action[None,:]
        if HALVED_POLICY:
            action = torch.cat([action, action], dim=1)
            last_qpos = torch.cat([last_qpos,last_qpos])
         #need to make the action shape for 2 robots, simply copy the action for the second robot even tho it wont be used
        if relative_action:
                # Subtract the action value of time_step t-1 from the action value of time_step t0, t1, ...
                # to get relative action values.
                absolute_start = actions[-1]
                action += absolute_start
        for i in range(t_act):
            t1 = time.time()  #
            current_pos = env.puppet_bot_left.arm.get_ee_pose()
            #add current joint states to the actions
            current_action = last_qpos + action[i]
            #.dxl.joint_states.position[:6]
            destination = mr.FKinSpace(env.puppet_bot_left.arm.robot_des.M, env.puppet_bot_left.arm.robot_des.Slist,
                                       current_action[0:6].detach().numpy())
            print(destination[0:3,3])
            print(current_pos[0:3,3])
            # We will have to findout what proper distances are
            # check the translation vector
            # if destination[2, 3] < 0.05:
            #     print("Warning: Height will be to low")
            #     while (True):
            #         # wait for the user to press enter
            #         if input("Press enter to continue") == "":
            #             break

            ts = env.step(current_action, move_time_arm=MOVE_TIME_ARM, move_time_gripper=MOVE_TIME_GRIPPER).observation
            if HALVED_POLICY:
                for key, value in ts.items():
                    if key != 'images' and key != 'parts_poses':
                        ts[key] = value[:value.shape[0] // 2]
            transform_last_image(ts, image_transforms)
            normalize_last_observation(ts)
            standardize_last_observation(ts)
            t2 = time.time()  #
            timesteps.append(ts)
            actions.append(current_action)
            observations.append(adjust_images(copy.deepcopy(ts)))
            actual_dt_history.append([t0, t1, t2])

    # save the data
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/observations/parts_poses': [],
        '/action': [],

    }
    for cam_name in CAMERA_NAMES:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts['qpos'])
        data_dict['/observations/qvel'].append(ts['qvel'])
        data_dict['/observations/effort'].append(ts['effort'])
        data_dict['/observations/parts_poses'].append(ts['parts_poses'])
        data_dict['/action'].append(action)
        for cam_name in CAMERA_NAMES:
            data_dict[f'/observations/images/{cam_name}'].append(ts['images'][cam_name])

    # HDF5
    t0 = time.time()
    index = get_auto_index(DATASET_DIR)
    dataset_path = DATASET_DIR + f'episode_{index}'
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in CAMERA_NAMES:
            _ = image.create_dataset(cam_name, (MAX_TIMESTEPS, 480, 640, 3), dtype='uint8',
                                     chunks=(1, 480, 640, 3), )
        number_joints = 14
        _ = obs.create_dataset('qpos', (MAX_TIMESTEPS, number_joints))
        _ = obs.create_dataset('qvel', (MAX_TIMESTEPS, number_joints))
        _ = obs.create_dataset('effort', (MAX_TIMESTEPS, number_joints))
        _ = obs.create_dataset('parts_poses', (MAX_TIMESTEPS, 7))
        _ = root.create_dataset('action', (MAX_TIMESTEPS, number_joints))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs')

def transform_last_image(observations,image_transforms):
    for key in IMAGE_KEYS:
        observations['images'][key] = image_transforms[key](torch.tensor(observations['images'][key]))

def normalize_last_observation(observations):
    for key in NORMALIZE_KEYS:
        # Normalize all trajectories
        observations[key] = normalize(torch.tensor(observations[key]), SCALER_VALUES[key], symmetric=NORMALIZE_SYMMETRICALLY)


def standardize_last_observation(observations):
    for key in STANDARDIZE_KEYS:
        # Standardize all trajectories
        observations[key] = standardize(torch.tensor(observations[key]), torch.tensor(SCALER_VALUES[key]))


def adjust_images(observation):
    # takes all images from the observation and transforms it to the correct shape, i.e. (#images, channels, ...)
    # for key in observations: #use key images on real stuff ['images']
    #     observations[key] = np.moveaxis(observations[key],-1,1) #dont forgetkey images

    # for cam_name in camera_names:
    #     observation['images'][cam_name] = np.moveaxis(observation['images'][cam_name],-1,0)[...,:CROP_SIZES[0],:CROP_SIZES[1]]
    # TODO normalize pixels to [0,1] by dividing by 255
    for key, value in observation.items():
        if key == 'images':
            # for cam_name in IMAGE_KEYS:
            #     observation[key][cam_name] = torch.tensor(value[cam_name])[...] 
            continue
        observation[key] = torch.tensor(value, dtype=torch.float32)[...]
    # for cam_name in IMAGE_KEYS:
    #     observation['images'][cam_name] = observation['images'][cam_name].type(torch.FloatTensor)
    return observation


def last_observations(observations):
    # take the last observations and put the into one dict of shape (1, t_obs, embedding_size)
    obs = dict()
    for key in observations[0].keys():
        if key == 'images':
            # for cam_name in IMAGE_KEYS:
            #     obs[cam_name] = torch.stack([observation[key][cam_name] for observation in observations])[None,...]
            continue
        obs[key] = torch.stack([observation[key] for observation in observations])[None, ...]

    return obs


def get_auto_index(dataset_dir, dataset_name_prefix='', data_suffix='hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def check_keys(normalize_keys, standardize_keys):
    if isinstance(standardize_keys, bool):
        if standardize_keys:
            standardize_keys = normalize_keys.copy()
        else:
            standardize_keys = []
    if isinstance(normalize_keys, bool):
        if normalize_keys:
            normalize_keys = standardize_keys.copy()
        else:
            normalize_keys = []
    return normalize_keys, standardize_keys


def read_config(hydra_config):
    global SCALER_VALUES, NORMALIZE_KEYS, STANDARDIZE_KEYS, NORMALIZE_IMAGES, CROP_SIZES, IMAGE_KEYS, RANDOM_CROP, PAD_START, PAD_END, NORMALIZE_SYMMETRICALLY
    SCALER_VALUES = OmegaConf.to_container(
        hydra_config['workspace_config']['env_config']['scaler_config']['scaler_values'])
    dataset_confs = OmegaConf.to_container(hydra_config['dataset_config'])
    NORMALIZE_IMAGES = hydra_config.dataset_config.get("normalize_images") if "normalize_images" in hydra_config.dataset_config else False
    CROP_SIZES = hydra_config.dataset_config.get("crop_sizes") if "crop_sizes" in hydra_config.dataset_config else []
    IMAGE_KEYS = hydra_config.dataset_config.get("cam_names") if "cam_names" in hydra_config.dataset_config else []
    RANDOM_CROP = hydra_config.dataset_config.get("random_crop") if "random_crop" in hydra_config.dataset_config else False
    PAD_START = hydra_config.dataset_config.get("pad_start") if "pad_start" in hydra_config.dataset_config else 0
    PAD_END = hydra_config.dataset_config.get("pad_end") if "pad_end" in hydra_config.dataset_config else 0
    NORMALIZE_SYMMETRICALLY = dataset_confs['normalize_symmetrically']
    normalize_keys = dataset_confs['normalize_keys']
    standardize_keys = dataset_confs['standardize_keys']
    normalize_keys, standardize_keys = check_keys(normalize_keys, standardize_keys)

    def check_for_action():
        global NORMALIZE_ACTION, STANDARDIZE_ACTION
        if 'action' in normalize_keys:
            normalize_keys.remove('action')
            NORMALIZE_ACTION = True
        else:
            NORMALIZE_ACTION = False
        if 'action' in standardize_keys:
            standardize_keys.remove('action')
            STANDARDIZE_ACTION = True
        else:
            STANDARDIZE_ACTION = False
        return normalize_keys, standardize_keys

    def convert_to_tensor():
        for key in SCALER_VALUES:
            for stat_value in SCALER_VALUES[key]:
                SCALER_VALUES[key][stat_value] = torch.tensor(SCALER_VALUES[key][stat_value])

    convert_to_tensor()
    NORMALIZE_KEYS, STANDARDIZE_KEYS = check_for_action()



def compose_image_transform(image_observation,image_key: str, image_size: Optional[List[int]], crop_size: Optional[List[int]], random_crop: bool, normalize: bool) -> Tuple[transforms.Compose, Tuple[int, int, int]]:
        image_transform_list = []

        if normalize:
            # Divide by 255 to scale to [0, 1]
            image_transform_list.append(transforms.Lambda(lambda x: x / 255.0))


        # Get tensor image
        tensor_image = image_observation[image_key][0]
        assert isinstance(tensor_image, torch.Tensor), f"Image must be a tensor, but is {type(tensor_image)}."
        # Determine original image shape
        org_image_shape = tuple(tensor_image.shape)
        assert len(org_image_shape) == 3, f"Image must have 3 dimensions, but has {len(org_image_shape)}."
        assert org_image_shape[0] == 3, f"Image must have 3 channels, but has {org_image_shape[0]} in shape {org_image_shape}."

        # Resizing images
        if image_size is None:
            image_shape = org_image_shape
        else:
            assert isinstance(image_size, list) and len(image_size) == 2
            image_shape = tuple([3] + image_size)
            if image_size != list(org_image_shape[1:]):
                image_transform_list.append(transforms.Resize(image_size, antialias=True))

        # (Random) cropping images
        if crop_size is not None:
            if not crop_size[0] <= image_shape[1] and crop_size[1] <= image_shape[2]:
                raise ValueError(f"Crop size {crop_size} is larger than image size {image_shape[1:]}.")
            image_shape = tuple([3] + crop_size)
            if random_crop:
                image_transform_list.append(transforms.RandomCrop(crop_size))
            else:
                image_transform_list.append(transforms.CenterCrop(crop_size))

        return transforms.Compose(image_transform_list), image_shape


if __name__ == "__main__":
    main()
