import numpy as np
from loguru import logger

from ..envs.legged_env import LeggedEnv
from ..command_manager.automaic_command_generator import AutomaticCommandGenerator

BASE_OBS = [
    # world frame
    "base_pos",
    "base_ori_euler_xyz",
    "base_ori_quat_wxyz",
    "world_lin_vel",
    "world_ang_vel",
    "world_lin_acc",
    "world_ang_acc",
    # base frame
    "base_lin_vel",
    "base_ang_vel",
    "base_lin_acc",
    "base_ang_acc",
    # sensor
    "base_lin_vel_sensor",
    "base_ang_vel_sensor",
    "base_lin_acc_sensor",
]
DOF_OBS = [
    "dof_pos",
    "dof_vel",
]
FEET_OBS = [
    "base_feet_pos",
    "base_feet_vel",
    "feet_vel",
    "feet_contact_state",
    "feet_contact_forces",
]


class RobotState:
    """
    noisy state: imu_acc, imu_omega, imu_quat, lin_vel, q, dq, tau, foot_contact_self
    ground truth state: base_position, base_quat, base_velocity, base_ang_velocity
    """

    def __init__(
        self, legged_env: LeggedEnv, command_generator: AutomaticCommandGenerator
    ):
        self.legged_env = legged_env
        self.command_generator = command_generator

        self.sim_env_properties, methods = list_sim_env_properties_methods(legged_env)
        logger.info(f"Properties of sim_env: {self.sim_env_properties}")
        self.command_properties, methods = list_sim_env_properties_methods(
            command_generator
        )
        logger.info(f"Properties of command_generator: {self.command_properties}")

    def reset(self):
        pass

    def get_state(self, state_name: str):
        if state_name in self.sim_env_properties:
            return getattr(self.legged_env, state_name)
        elif state_name in self.command_properties:
            return getattr(self.command_generator, state_name)
        else:
            logger.error(f"Cannot find state: {state_name}")
            return None


def list_sim_env_properties_methods(obj):
    attrs = dir(obj.__class__)  # Change to inspect the class
    properties = [
        attr
        for attr in attrs
        if isinstance(getattr(obj.__class__, attr, None), property)
    ]
    methods = [
        attr
        for attr in attrs
        if callable(getattr(obj.__class__, attr, None)) and not attr.startswith("__")
    ]
    return properties, methods
