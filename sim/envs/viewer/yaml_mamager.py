import os.path as osp
import yaml
from loguru import logger


# yaml
def save_to_yaml(config, filename="sim_config.yml"):
    with open(osp.join(osp.dirname(__file__), filename), "w") as f:
        yaml.dump(config, f)
    logger.info(f"Saved config to {filename}, dict={config}")


def load_from_yaml(filename="sim_config.yml"):
    with open(osp.join(osp.dirname(__file__), filename), "r") as f:
        parsed_dict = yaml.safe_load(f)
    logger.info(f"Loaded config from {filename}, dict={parsed_dict}")
    return parsed_dict
