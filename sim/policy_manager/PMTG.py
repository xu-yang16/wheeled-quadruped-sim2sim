# SPDX-FileCopyrightText: Copyright (c) 2023, HUAWEI TECHNOLOGIES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Sequence
import torch
import numpy as np

from .math_utils import (
    get_euler_xyz,
    coordinate_rotation,
    to_torch,
    quat_apply,
    torch_rand,
)

_TROT_PHASE_OFFSET = [0, 0.5, 0.5, 0]
_TROT_DUTY = 0.4
_TROT_F = 1.2

_WALK_PHASE_OFFSET = [0, 0.25, 0.5, 0.75]
_WALK_DUTY = 0.225
_WALK_F = 0.8

_DRIVING_PHASE_OFFSET = [0.0, 0.0, 0.0, 0.0]
_DRIVING_DUTY = 0.0
_DRIVING_F = 0.5


class PMTrajectoryGenerator:
    """
    Class for generating foot target trajectories for a quadruped robot.
    """

    def __init__(
        self,
        device: torch.device,
        num_envs: int,
        param: Any,
    ):
        """
        Initialize the PMTrajectoryGenerator.

        Args:
            clock: The clock object for timing.
            device: The device to run the calculations on.
            num_envs: The number of parallel environments.
            param: The parameters for trajectory generation.

        Raises:
            Exception: If an invalid task_name is provided.
        """
        UPPER_LEG_LENGTH, LOWER_LEG_LENGTH, HIP_LENGTH = 0.213, 0.213, 0.08

        HIP_POSITION = np.array(
            [
                [0.203, -0.142, 0],
                [0.203, 0.142, 0],
                [-0.156, -0.142, 0],
                [-0.156, 0.142, 0],
            ]
        )
        COM_OFFSET = np.array([0.0, 0.0, 0.0])
        HIP_OFFSETS = (
            np.array(
                [
                    [0.1881, -0.04675, 0.0],
                    [0.1881, 0.04675, 0.0],
                    [-0.1881, -0.04675, 0.0],
                    [-0.1881, 0.04675, 0.0],
                ]
            )
            + COM_OFFSET
        )  # FL, FR, RL, RR
        # Store robot parameters
        self.UPPER_LEG_LENGTH = UPPER_LEG_LENGTH
        self.LOWER_LEG_LENGTH = LOWER_LEG_LENGTH
        self.HIP_LENGTH = HIP_LENGTH
        self.HIP_POSITION = HIP_POSITION
        self.COM_OFFSET = COM_OFFSET
        self.HIP_OFFSETS = HIP_OFFSETS

        self.device = device
        self.num_envs = num_envs

        self.max_clearance = param.max_clearance
        self.body_height = param.body_height
        self.max_horizontal_offset = param.max_horizontal_offset
        self.train_mode = param.train_mode
        self.gait_type = param.gait_type

        # Set initial phase based on gait type
        # for playing
        self.initial_phase = to_torch(_TROT_PHASE_OFFSET, device=self.device).repeat(
            self.num_envs, 1
        )
        self.duty_factor = 0.4 * torch.ones(self.num_envs, 1, device=self.device)
        self.duty_factor_tensor = self.duty_factor.repeat(1, 4)
        base_frequency = 1.2
        self.base_frequency_tensor = base_frequency * torch.ones(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )

        self.default_joint_position = torch.zeros(
            self.num_envs, 12, dtype=torch.float, device=self.device
        )

        # Initial the joint positions and joint angles
        self.foot_target_position_in_hip_frame = torch.zeros(
            self.num_envs, 12, dtype=torch.float, device=self.device
        )
        self.foot_target_position_in_base_frame = torch.zeros(
            self.num_envs, 12, dtype=torch.float, device=self.device
        )
        self.target_joint_angles = torch.zeros(
            self.num_envs, 12, dtype=torch.float, device=self.device
        )

        self.is_swing = torch.zeros(
            (self.num_envs, 4), dtype=torch.bool, device=self.device
        )

        # FR, FL, RR, RL
        self.l_hip_sign = torch.tensor(
            [-1, 1, -1, 1], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)

        self.phi = self.initial_phase.clone()
        self.swing_phi = torch.zeros_like(self.phi)
        self.delta_phi = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device
        )
        self.cos_phi = torch.cos(self.phi * 2 * torch.pi)
        self.sin_phi = torch.sin(self.phi * 2 * torch.pi)
        self.reset_time = torch.tensor(
            [0.0], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.time_since_reset = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )

        self.foot_trajectory = torch.zeros(
            (self.num_envs, 4, 3), dtype=torch.float, device=self.device
        )
        self.foot_trajectory_z = (
            torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
            - self.body_height
        )

        self.com_offset = to_torch(COM_OFFSET, device=self.device)
        self.hip_offsets = to_torch(HIP_OFFSETS, device=self.device)  # FL, FR, RL, RR
        self.hip_position = to_torch(HIP_POSITION, device=self.device)  # FL, FR, RL, RR

        self.f_up = self.gen_func(param.z_updown_height_func[0])
        self.f_down = self.gen_func(param.z_updown_height_func[1])

    def randomize_gait(self, index_list):
        length_list = len(index_list)
        if length_list == 0:
            return
        if not self.train_mode:
            return

        if self.gait_type == "trot":
            # [0.4~0.6, 0.4~0.6, -0.1~0.1]
            # duty factor: 0.3~0.5
            # frequency: 0.8~1.6
            self.initial_phase[index_list, :] = to_torch(
                _TROT_PHASE_OFFSET, device=self.device
            ).repeat(length_list, 1)

            # FIXME: remove random phase
            # self.duty_factor[index_list, :] = 0.5
            # self.duty_factor_tensor = self.duty_factor.repeat(1, 4)
            # self.base_frequency_tensor[index_list, :] = 1.2
            self.initial_phase[index_list, :] += torch_rand(
                0.0, 1.0, length_list, 1, device=self.device
            )  # random initial FL phase
            self.initial_phase[index_list, 1:] += torch_rand(
                -0.1, 0.1, length_list, 3, device=self.device
            )
            self.initial_phase = self.initial_phase % 1
            self.duty_factor[index_list, :] = torch_rand(
                0.3, 0.5, length_list, 1, device=self.device
            )
            self.duty_factor_tensor = self.duty_factor.repeat(1, 4)
            self.base_frequency_tensor[index_list, :] = torch_rand(
                0.8, 1.6, length_list, 1, device=self.device
            )
        elif self.gait_type == "walk":
            # [0.2~0.3, 0.45~0.55, 0.7~0.8]
            # duty factor: 0.2~0.25
            # frequency: 0.6~1.0
            self.initial_phase[index_list, :] = to_torch(
                _WALK_PHASE_OFFSET, device=self.device
            )
            self.initial_phase[index_list, :] += torch_rand(
                0.0, 1.0, length_list, 1, device=self.device
            )
            self.initial_phase[index_list, 1:] += torch_rand(
                -0.05, 0.05, length_list, 3, device=self.device
            )
            self.initial_phase = self.initial_phase % 1
            self.duty_factor[index_list, :] = torch_rand(
                0.2, 0.25, length_list, 1, device=self.device
            )
            self.duty_factor_tensor = self.duty_factor.repeat(1, 4)
            self.base_frequency_tensor[index_list, :] = torch_rand(
                0.6, 1.0, length_list, 1, device=self.device
            )
        elif self.gait_type == "driving":
            # [0.0~1.0, 0.0~1.0, 0.0~1.0]
            # duty factor: 0.0~0.0
            # frequency: 0.0~1.0
            self.initial_phase[index_list, :] = to_torch(
                _DRIVING_PHASE_OFFSET, device=self.device
            )
            self.initial_phase[index_list, :] += torch_rand(
                0.0, 1.0, length_list, 1, device=self.device
            )
            self.initial_phase[index_list, 1:] += torch_rand(
                0.0, 1.0, length_list, 3, device=self.device
            )
            self.initial_phase = self.initial_phase % 1
            self.duty_factor[index_list, :] = torch.zeros(
                length_list, 1, device=self.device
            )
            self.duty_factor_tensor = self.duty_factor.repeat(1, 4)
            self.base_frequency_tensor[index_list, :] = torch_rand(
                0.0, 1.0, length_list, 1, device=self.device
            )
        elif self.gait_type == "hybrid":
            # TODO:hybrid sampling
            pass

    def set_gait(self, index_list):
        length_list = len(index_list)
        if length_list == 0:
            return

        if self.gait_type == "trot":
            # [0.4~0.6, 0.4~0.6, -0.1~0.1]
            # duty factor: 0.3~0.5
            # frequency: 0.8~1.6
            self.initial_phase[index_list, :] = to_torch(
                _TROT_PHASE_OFFSET, device=self.device
            ).repeat(length_list, 1)
            self.initial_phase = self.initial_phase % 1
            self.duty_factor[index_list, :] = 0.4
            self.duty_factor_tensor = self.duty_factor.repeat(1, 4)
            self.base_frequency_tensor[index_list, :] = 1.2
        elif self.gait_type == "walk":
            # [0.2~0.3, 0.45~0.55, 0.7~0.8]
            # duty factor: 0.2~0.25
            # frequency: 0.6~1.0
            self.initial_phase[index_list, :] = to_torch(
                _WALK_PHASE_OFFSET, device=self.device
            )
            self.duty_factor[index_list, :] = 0.225
            self.duty_factor_tensor = self.duty_factor.repeat(1, 4)
            self.base_frequency_tensor[index_list, :] = 0.8
        elif self.gait_type == "driving":
            # [0.0~1.0, 0.0~1.0, 0.0~1.0]
            # duty factor: 0.0~0.0
            # frequency: 0.0~1.0
            self.initial_phase[index_list, :] = to_torch(
                _DRIVING_PHASE_OFFSET, device=self.device
            )
            self.duty_factor[index_list, :] = torch.zeros(
                length_list, 1, device=self.device
            )
            self.duty_factor_tensor = self.duty_factor.repeat(1, 4)
            self.base_frequency_tensor[index_list, :] = 0.5

    def gen_func(self, func_name):
        """
        Generate a lambda function based on the provided function name.

        Args:
            func_name (str): Name of the function to generate.

        Returns:
            function: Lambda function based on the provided function name.

        Raises:
            NotImplementedError: If the provided function name is not supported.
        """
        if func_name == "cubic_up":
            return lambda x: -16 * x**3 + 12 * x**2
        elif func_name == "cubic_down":
            return lambda x: 16 * x**3 - 36 * x**2 + 24 * x - 4
        elif func_name == "linear_down":
            return lambda x: 2.0 - 2.0 * x
        elif func_name == "sin":
            return lambda x: torch.sin(torch.pi * x)
        else:
            raise NotImplementedError("PMTG z height")

    def reset(self, index_list, current_time):
        """
        Reset the specified indices in the motion planner.

        Args:
            index_list (list): A list of indices to reset.
        """

        self.set_gait(index_list)
        # self.randomize_gait(index_list)
        self.phi[index_list] = self.initial_phase[index_list]
        self.swing_phi[index_list] = 0
        self.delta_phi[index_list] = 0
        self.cos_phi[index_list] = torch.cos(self.phi[index_list] * 2 * torch.pi)
        self.sin_phi[index_list] = torch.sin(self.phi[index_list] * 2 * torch.pi)
        self.is_swing[index_list] = False
        self.reset_time[index_list] = current_time
        self.time_since_reset[index_list] = 0

    def update_observation(self):
        """
        Update the last action and parameters of the Central Pattern Generator (CPG) and return the observation.

        Returns:
            observation (torch.Tensor): The updated observation containing delta_phi, cos_phi, sin_phi, and base_frequency.

        """
        observation = torch.cat(
            (
                self.delta_phi,  # 4
                self.cos_phi,  # 4
                self.sin_phi,  # 4
                self.base_frequency_tensor,  # 1
                self.duty_factor,  # 1
            ),
            dim=1,
        )

        return observation

    def get_foot_positions(self):
        return self.foot_target_position_in_base_frame

    def get_action(
        self, delta_phi, residual_xyz, residual_angle, base_orientation, current_time
    ):
        """
        compute the position in base frame, given the base orientation.

        Args:
          delta_phi: phase variable.
          residual_xyz: residual in horizontal hip reference frame.
          residual_angle: residual in joint space.
          base_orientation: quaternion (w,x,y,z) of the base link.

        Returns:
            target_joint_angles: joint angle of for leg (FL,FR,RL,RR)
        """
        delta_phi, residual_angle = delta_phi.to(self.device), residual_angle.to(
            self.device
        )
        self.gen_foot_target_position_in_horizontal_hip_frame(
            delta_phi, residual_xyz, current_time
        )
        self.foot_target_position_in_base_frame = self.transform_to_base_frame(
            self.foot_target_position_in_hip_frame, base_orientation
        )
        self.target_joint_angles = self.get_target_joint_angles(
            self.foot_target_position_in_base_frame
        )
        self.target_joint_angles += residual_angle

        return self.target_joint_angles

    def gen_foot_trajectory_axis_z(
        self, delta_phi: Sequence[float], current_time: float
    ) -> Sequence[float]:
        """
        Generate the foot trajectory along the z-axis.

        Args:
            delta_phi: A sequence of floats representing the change in phase for each leg.
            t: The current time.

        Returns:
            A sequence of floats representing the foot trajectory along the z-axis.
        """

        self.time_since_reset = current_time - self.reset_time
        self.phi = (
            self.initial_phase
            + self.base_frequency_tensor * self.time_since_reset
            + delta_phi
        ) % 1

        self.delta_phi = delta_phi
        self.cos_phi = torch.cos(self.phi * 2 * torch.pi)
        self.sin_phi = torch.sin(self.phi * 2 * torch.pi)

        if self.gait_type == "driving":
            self.foot_trajectory[:, :, 2] = -self.body_height
            self.foot_trajectory[:, :, 0] = -self.max_horizontal_offset
        else:
            k3 = self.phi < self.duty_factor_tensor
            self.is_swing = k3.clone()
            self.swing_phi = self.phi / self.duty_factor  # [0,1)
            factor = torch.where(
                self.swing_phi < 0.5,
                self.f_up(self.swing_phi),
                self.f_down(self.swing_phi),
            )
            self.foot_trajectory[:, :, 2] = (
                factor * (self.is_swing * self.max_clearance) - self.body_height
            )
            self.foot_trajectory[:, :, 0] = (
                -self.max_horizontal_offset
                * torch.sin(self.swing_phi * 2 * torch.pi)
                * self.is_swing
            )

    def gen_foot_target_position_in_horizontal_hip_frame(
        self,
        delta_phi: Sequence[float],
        residual_xyz: Sequence[float],
        current_time: float,
    ) -> Sequence[float]:
        """
        Compute the foot target positions in the horizontal hip reference frame.

        Args:
            delta_phi: A sequence of floats representing the phase variable.
            residual_xyz: A sequence of floats representing the residual in the horizontal hip reference frame.

        Returns:
            A sequence of floats representing the foot target positions in the horizontal hip reference frame.
        """
        self.foot_target_position_in_hip_frame = residual_xyz.reshape(-1, 4, 3)
        self.gen_foot_trajectory_axis_z(delta_phi, current_time)
        self.foot_target_position_in_hip_frame += self.foot_trajectory

        return self.foot_target_position_in_hip_frame

    def transform_to_base_frame(self, position, quaternion):
        """
        Compute the position in the base frame, given the base orientation.

        Args:
            position: A tensor representing the point position.
            quaternion: A tensor representing the quaternion (x, y, z, w) of the base link.

        Returns:
            A tensor representing the position in the base frame.
        """
        # return position + self.hip_position.unsqueeze(0)
        rpy = get_euler_xyz(quaternion)
        rpy[:, 2] = 0
        R = torch.matmul(
            coordinate_rotation(0, rpy[:, 0]), coordinate_rotation(1, rpy[:, 1])
        )
        rotated_position = torch.matmul(R, position.float().transpose(1, 2)).transpose(
            1, 2
        )
        rotated_position = rotated_position + self.hip_position.unsqueeze(0)

        return rotated_position

    def quat_apply_feet_positions(self, quat, positions):
        """
        Apply quaternion rotation to the foot positions.

        Args:
            quat: A tensor representing the quaternion (x, y, z, w).
            positions: A tensor representing the foot positions.

        Returns:
            A tensor representing the foot positions after applying the quaternion rotation.
        """
        quat *= torch.tensor([-1, -1, -1, 1]).to(self.device)
        num_feet = positions.shape[1]
        quat = quat.repeat(1, num_feet).reshape(-1, num_feet)
        quat_pos = quat_apply(quat, positions.reshape(-1, 3))

        return quat_pos.view(positions.shape)

    def get_target_joint_angles(self, target_position_in_base_frame):
        """
        Compute the joint angles given the foot target positions in the base frame.

        Args:
            target_position_in_base_frame: A tensor representing the foot target positions in the base frame.

        Returns:
            A tensor representing the joint angles for each leg.
        """
        foot_position = target_position_in_base_frame - self.hip_offsets
        joint_angles = self.foot_position_in_hip_frame_to_joint_angle(foot_position)

        return joint_angles

    def foot_position_in_hip_frame_to_joint_angle(self, foot_position):
        """
        Compute the motor angles for one leg using inverse kinematics (IK).

        Args:
            foot_position: A tensor representing the foot positions in the hip frame.

        Returns:
            A tensor representing the motor angles for one leg.
        """
        l_up = self.UPPER_LEG_LENGTH
        l_low = self.LOWER_LEG_LENGTH
        l_hip = self.HIP_LENGTH * self.l_hip_sign
        x, y, z = foot_position[:, :, 0], foot_position[:, :, 1], foot_position[:, :, 2]
        theta_knee_input = torch.clip(
            (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) / (2 * l_low * l_up),
            -1,
            1,
        )
        theta_knee = -torch.arccos(theta_knee_input)
        l = torch.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * torch.cos(theta_knee))
        theta_hip_input = torch.clip(-x / l, -1, 1)
        theta_hip = torch.arcsin(theta_hip_input) - theta_knee / 2
        c1 = l_hip * y - l * torch.cos(theta_hip + theta_knee / 2) * z
        s1 = l * torch.cos(theta_hip + theta_knee / 2) * y + l_hip * z
        theta_ab = torch.atan2(s1, c1)

        res = torch.cat(
            (theta_ab.unsqueeze(2), theta_hip.unsqueeze(2), theta_knee.unsqueeze(2)),
            dim=2,
        ).reshape(foot_position.shape[0], -1)
        return res
