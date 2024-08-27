import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from trajectory_diffusion.utils.setup_helper import setup_agent, parse_wandb_to_hydra_config

from utils.robot_tester import RobotTester

OmegaConf.register_new_resolver("eval", eval)


CONFIG = "test_trained_agent_in_env_furniture"
CONFIG_PATH = "/home/studentgroup1/trajectory-diffusion-prak/conf"


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG)
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
        hydra_config = OmegaConf.merge(hydra_config, cfg.config_to_change)

    # Load and update robot_config with new values
    robot_config = OmegaConf.load(cfg.robot_config_yaml)
    if cfg.get("robot_config_to_change") is not None:
        robot_config = OmegaConf.merge(robot_config, cfg.robot_config_to_change)
    robot_config.hydra_config = hydra_config
    robot_config.agent_name = cfg.agent_name
    robot_config_params = OmegaConf.to_container(robot_config, resolve=True)

    agent = setup_agent(hydra_config)
    agent.load_pretrained(cfg.weights)

    robot_tester = RobotTester(robot_config_params)

    test_results = robot_tester.test_agent(agent)

    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
