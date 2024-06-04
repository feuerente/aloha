import hydra
import torch
import logging
import h5py

from omegaconf import DictConfig, OmegaConf
import numpy as np
import random

from trajectory_diffusion.utils.setup_helper import setup_agent_and_workspace, parse_wandb_to_hydra_config
#from real_env import make_real_env, get_action
from fake_env import make_real_env
from tqdm import tqdm
import time

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("eval", eval)

CONFIG = "experiments/real_furniture_bench/train_dp_transformer_image.yaml"
CONFIG = "test_trained_agent_in_env_furniture"
DATASET_PATH = './rollout_1'
MAX_TIMESTEPS = 10
@hydra.main(version_base=None, config_path="../../trajectory-diffusion-prak/conf", config_name=CONFIG)
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
    #hydra_config = OmegaConf.merge(hydra_config, cfg.to_change)

    # Setup agent, and workspace
    agent, workspace = setup_agent_and_workspace(hydra_config)

    # Load the weights
    agent.load_pretrained(cfg.weights)

    #agent loaded, now we can test it
    #test_results = workspace.test_agent(agent, cfg.num_trajectories)
    #print(f"Test results: {test_results}")
    
    #prepare the real environment
    env = make_real_env(init_node=False, setup_robots=False)
    # Data collection
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []
    observations = [adjust_images(ts)]#use ts.observation on real_env
    
    for t in tqdm(range(MAX_TIMESTEPS)):
        t0 = time.time() #
        action = agent.predict(observation=observations[-1],extra_inputs=dict())
        t1 = time.time() #
        ts = env.step(action)
        t2 = time.time() #
        timesteps.append(ts)
        actions.append(action)
        observations.append(adjust_images(ts))
        actual_dt_history.append([t0, t1, t2])
    
   
def adjust_images(observations):
    #takes all images from the observation and transforms it to the correct shape, i.e. (#images, channels, ...)
    # for key in observations: #use key images on real stuff ['images']
    #     observations[key] = np.moveaxis(observations[key],-1,1) #dont forgetkey images
    return np.moveaxis(observations['cam_low'],-1,1)#observations


if __name__ == "__main__":
    main()


