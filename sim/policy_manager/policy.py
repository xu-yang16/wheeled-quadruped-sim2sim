import os.path as osp
import numpy as np
import torch
from loguru import logger

from .PMTG import PMTrajectoryGenerator
from .policy_config.my_legged_robot_config import MyLeggedRobotCfg

# for type clarification
from ..state_manager.state import RobotState
from ..common.utils import MotorCommand


class Policy:
    """
    Mapping from observation to action:
    - load policy from model_path
    - observation history management and scaling
    - action scaling
    """

    def __init__(
        self,
        robot_state: RobotState,
        model_path,
        device="cpu",
    ):
        self.robot_state = robot_state
        # load RL policy
        logger.info(f"Loading policy from {model_path}")
        self.device = device
        self.policy = load_policy(osp.join(osp.dirname(__file__), model_path))

        # config
        self.cfg = MyLeggedRobotCfg

        # dimensions
        self.num_actor_obs = self.cfg.env.num_actor_observations
        self.num_obs_history = self.cfg.env.num_observation_history
        self.num_actions = 12 + 4  # 12 leg actions + 4 wheel actions

        # record obs and obs history
        self.obs_numpy = np.zeros(self.num_actor_obs)
        self.obs_history_numpy = np.zeros((self.num_obs_history, self.num_actor_obs))
        self.action_numpy = np.zeros(self.num_actions)
        self.last_action_numpy = np.zeros(self.num_actions)
        # whether to use ground truth obs

        # scale and decode
        self.pmtg = PMTrajectoryGenerator(
            device=self.device, num_envs=1, param=self.cfg.pmtg
        )
        self.default_dof_pos = self.load_default_dof_pos(self.cfg)
        self.motor_command = MotorCommand(
            joint_names=[
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
            ]
            + [
                "FR_wheel_joint",
                "FL_wheel_joint",
                "RR_wheel_joint",
                "RL_wheel_joint",
            ],
            target_pos=np.zeros(16),
            target_vel=np.zeros(16),
            Kps=np.zeros(16),
            Kds=np.zeros(16),
        )

    def load_default_dof_pos(self, cfg):
        joint_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]
        return np.array(
            [cfg.init_state.default_joint_angles[name] for name in joint_names]
        )

    def reset(self):
        sim_time = self.robot_state.get_state("sim_time")
        self.pmtg.reset(index_list=[0], current_time=sim_time)

    def update(self):
        """
        Need to call `construct_obs` before calling this function.
        - Scale observation and observation history.
        - Update action based on observation and observation history.
        - Decode action to motor_cmd.
        """

        # update obs from state
        base_lin_vel, base_ang_vel, projected_gravity, dof_pos, dof_vel = (
            self.robot_state.get_state("base_lin_vel"),
            self.robot_state.get_state("base_ang_vel"),
            self.robot_state.get_state("projected_gravity"),
            self.robot_state.get_state("dof_pos"),
            self.robot_state.get_state("dof_vel"),
        )
        # remove wheel pos from dof_pos
        dof_pos = np.delete(dof_pos, [3, 7, 11, 15])
        current_commands = self.robot_state.get_state("current_commands")

        # construct obs
        scaled_obs_numpy, scaled_obs_history_numpy = self.construct_scaled_obs(
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            dof_pos,
            dof_vel,
            current_commands,
        )
        self.obs_numpy, self.obs_history_numpy = (
            scaled_obs_numpy,
            scaled_obs_history_numpy,
        )
        # encode obs
        scaled_obs_torch, scaled_obs_history_torch = self.encode_scaled_obs(
            scaled_obs_numpy, scaled_obs_history_numpy
        )

        # update action with the policy
        policy_outputs_torch = self.policy(scaled_obs_torch, scaled_obs_history_torch)

        # decode action to motor command
        quat_xyzw = self.robot_state.get_state("world_quat_xyzw")
        leg_actions, wheel_actions, actions = self.decode_action(
            policy_outputs_torch,
            self.robot_state.get_state("world_quat_xyzw"),
            self.robot_state.get_state("sim_time"),
        )

        self.last_action_numpy = self.action_numpy.copy()
        self.action_numpy = actions.copy()

        self.motor_command.target_pos[:12] = leg_actions
        self.motor_command.target_pos[12:] = 0.0

        self.motor_command.target_vel[:12] = 0.0
        self.motor_command.target_vel[12:] = wheel_actions

        leg_kp, leg_kd = self.leg_kp_kd
        wheel_kp, wheel_kd = self.wheel_kp_kd
        self.motor_command.Kps[:12] = leg_kp
        self.motor_command.Kps[12:] = wheel_kp
        self.motor_command.Kds[:12] = leg_kd
        self.motor_command.Kds[12:] = wheel_kd

        return self.motor_command

    def construct_scaled_obs(
        self,
        base_lin_vel: np.array,
        base_ang_vel: np.array,
        projected_gravity: np.array,
        dof_pos: np.array,
        dof_vel: np.array,
        user_cmds: np.array,
    ):
        cpg_phase_info = self.pmtg.update_observation().detach().numpy().squeeze()

        obs_scales = self.cfg.observation.obs_scales
        command_scale = np.array(
            [obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel]
        )

        scaled_dof_vel = dof_vel * obs_scales.dof_vel
        scaled_dof_vel[[3, 7, 11, 15]] *= obs_scales.wheel_dof_vel

        raw_obs_numpy = np.concatenate(
            (
                base_ang_vel * obs_scales.ang_vel,  # 3
                projected_gravity,  # 3
                (dof_pos - self.default_dof_pos) * obs_scales.dof_pos,  # 12
                scaled_dof_vel,  # 16
                cpg_phase_info,  # 14
                self.action_numpy,  # 16
                self.last_action_numpy,  # 16
                user_cmds * command_scale,  # 3
            )
        )
        if np.all(self.obs_history_numpy) == 0:
            self.obs_history_numpy = np.tile(raw_obs_numpy, (self.num_obs_history, 1))
        else:
            self.obs_history_numpy = np.roll(self.obs_history_numpy, -1, axis=0)
            self.obs_history_numpy[-1] = raw_obs_numpy
        return raw_obs_numpy, self.obs_history_numpy

    def encode_scaled_obs(
        self, scaled_obs_numpy: np.array, scaled_obs_history_numpy: np.array
    ):
        """
        - Encode obs_numpy to obs_torch.
        """
        scaled_obs_torch = (
            torch.from_numpy(scaled_obs_numpy).to(self.device).reshape(1, -1)
        )
        scaled_obs_history_torch = (
            torch.from_numpy(scaled_obs_history_numpy).to(self.device).reshape(1, -1)
        )
        # scale obs and obs history
        return scaled_obs_torch, scaled_obs_history_torch

    def decode_action(
        self, policy_outputs_torch: torch.tensor, quat_xyzw: np.array, sim_time: float
    ):
        """
        Decode action to motor_cmd.
        Params:
         - policy_outputs_numpy: torch sensor of shape (1, num_actions)
        Returns:
            leg_actions: np.array of shape (12,)
            wheel_actions: np.array of shape (4,)
            actions: np.array of shape (16,)
        """
        delta_phi = policy_outputs_torch[:, 0:4] * self.cfg.pmtg.delta_phi_scale
        delta_phi = torch.clip(
            delta_phi, -self.cfg.pmtg.max_delta_phi, self.cfg.pmtg.max_delta_phi
        )

        residual_xyz = policy_outputs_torch[:, 4:16] * self.cfg.pmtg.residual_xyz_scale
        residual_xyz = torch.clip(
            residual_xyz,
            -self.cfg.pmtg.max_residual_xyz,
            self.cfg.pmtg.max_residual_xyz,
        )

        residual_angle = torch.zeros(1, 12)
        base_quat = torch.from_numpy(quat_xyzw).unsqueeze(0)  # x, y, z, w
        pmtg_joints = self.pmtg.get_action(
            delta_phi, residual_xyz, residual_angle, base_quat, sim_time
        )
        wheel_vel = policy_outputs_torch[:, 16:20] * self.cfg.control.wheel_action_scale

        clip_actions = self.cfg.control.clip_actions
        clip_wheel_actions = self.cfg.control.clip_wheel_actions
        leg_actions = torch.clip(pmtg_joints, -clip_actions, clip_actions)
        wheel_actions = torch.clip(wheel_vel, -clip_wheel_actions, clip_wheel_actions)

        actions = (
            torch.cat([leg_actions, wheel_actions], dim=1).squeeze(0).detach().numpy()
        )

        return (
            leg_actions.squeeze(0).detach().numpy(),
            wheel_actions.squeeze(0).detach().numpy(),
            actions,
        )

    @property
    def leg_kp_kd(self):
        return self.cfg.control.leg_stiffness, self.cfg.control.leg_damping

    @property
    def wheel_kp_kd(self):
        return 0, 2.0


def load_policy(
    logdir,
    actor_model="actor.jit",
    adaptation_model="adaptation_module.jit",
    device="cpu",
):
    body = torch.jit.load(osp.join(logdir, actor_model))
    adaptation_module = torch.jit.load(osp.join(logdir, adaptation_model))

    def policy(actor_obs, obs_history):
        latent = adaptation_module.forward(
            obs_history.float()
            .reshape(
                -1,
            )
            .to(device)
            .unsqueeze(0)
        )
        action = body.forward(torch.cat((actor_obs.float().to(device), latent), dim=-1))
        return action

    return policy
