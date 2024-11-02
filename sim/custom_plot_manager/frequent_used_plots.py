import numpy as np
from ..common.utils import MotorCommand
from ..state_manager.state import RobotState
from ..policy_manager.policy import Policy


class LeggedPlotManager:
    def __init__(self):
        self.plot_manager_list = []

    def add_plot_manager(self, plot_manager):
        self.plot_manager_list.append(plot_manager)

    def reset(self):
        for plot_manager in self.plot_manager_list:
            plot_manager.reset()

    def update(
        self,
        robot_state: RobotState,
        motor_cmd: MotorCommand,
        policy: Policy,
        sim_time: float,
    ):
        if len(self.plot_manager_list) == 0:
            return
        self.add_base_data(robot_state, sim_time)
        self.add_joint_pos_data(robot_state, motor_cmd, sim_time)
        self.add_joint_vel_data(robot_state, motor_cmd, sim_time)

        self.add_joint_torque_data(robot_state, sim_time)

        self.add_feet_state_data(robot_state, sim_time)
        self.add_feet_contact_data(robot_state, sim_time)
        self.add_policy_data(policy, sim_time)

    def add_base_data(self, robot_state: RobotState, sim_time: float):
        actual_vel = np.array(
            [
                robot_state.get_state("base_lin_vel")[0],
                robot_state.get_state("base_lin_vel")[1],
                robot_state.get_state("base_lin_vel")[2],
                robot_state.get_state("base_lin_vel")[0],
                robot_state.get_state("base_ang_vel")[1],
                robot_state.get_state("base_ang_vel")[2],
            ]
        )
        commanded_vel = np.array(
            [
                robot_state.get_state("current_commands")[0],
                robot_state.get_state("current_commands")[1],
                0,
                0,
                0,
                robot_state.get_state("current_commands")[2],
            ]
        )
        # 记录 Commanded Velocity Tracking
        for row_idx, title in enumerate(
            [
                "lin_vel_x",
                "lin_vel_y",
                "lin_vel_z",
                "ang_vel_x",
                "ang_vel_y",
                "ang_vel_z",
            ]
        ):
            for plot_manager in self.plot_manager_list:
                plot_manager.log(
                    f"Vel_Tracking/{title}/Commanded", commanded_vel[row_idx]
                )
                plot_manager.log(f"Vel_Tracking/{title}/Actual", actual_vel[row_idx])

    def add_joint_pos_data(
        self, robot_state: RobotState, motor_cmd: MotorCommand, sim_time: float
    ):
        dof_pos = robot_state.get_state("dof_pos")
        desired_dof_pos = motor_cmd.target_pos

        dof_names = robot_state.get_state("dof_names")
        desired_dof_names = motor_cmd.joint_names
        # 记录 Joint Position Tracking
        for leg_name in ["FR", "FL", "RR", "RL"]:
            for part_name in ["hip", "thigh", "calf", "wheel"]:
                desired_idx = desired_dof_names.index(f"{leg_name}_{part_name}_joint")
                dof_idx = dof_names.index(f"{leg_name}_{part_name}_joint")
                for plot_manager in self.plot_manager_list:
                    plot_manager.log(
                        f"Joint_Position_Tracking/{leg_name}_{part_name}/Commanded",
                        desired_dof_pos[desired_idx],
                    )
                    plot_manager.log(
                        f"Joint_Position_Tracking/{leg_name}_{part_name}/Actual",
                        dof_pos[dof_idx],
                    )

    def add_joint_vel_data(
        self, robot_state: RobotState, motor_cmd: MotorCommand, sim_time: float
    ):
        dof_vel = robot_state.get_state("dof_vel")
        desired_dof_vel = motor_cmd.target_vel

        dof_names = robot_state.get_state("dof_names")
        desired_dof_names = motor_cmd.joint_names
        # 记录 Joint Velocity Tracking
        for leg_name in ["FR", "FL", "RR", "RL"]:
            for part_name in ["hip", "thigh", "calf", "wheel"]:
                desired_idx = desired_dof_names.index(f"{leg_name}_{part_name}_joint")
                dof_idx = dof_names.index(f"{leg_name}_{part_name}_joint")
                for plot_manager in self.plot_manager_list:
                    plot_manager.log(
                        f"Joint_Velocity_Tracking/{leg_name}_{part_name}/Commanded",
                        desired_dof_vel[desired_idx],
                    )
                    plot_manager.log(
                        f"Joint_Velocity_Tracking/{leg_name}_{part_name}/Actual",
                        dof_vel[dof_idx],
                    )

    def add_joint_torque_data(self, robot_state: RobotState, sim_time: float):
        dof_torque = robot_state.get_state("dof_torque")
        dof_names = robot_state.get_state("dof_names")
        # 记录 Joint Velocity Tracking
        for leg_name in ["FR", "FL", "RR", "RL"]:
            for part_name in ["hip", "thigh", "calf", "wheel"]:
                idx = dof_names.index(f"{leg_name}_{part_name}_joint")

                for plot_manager in self.plot_manager_list:
                    plot_manager.log(
                        f"Joint_Torque_Tracking/{leg_name}_{part_name}",
                        dof_torque[idx],
                    )

    def add_feet_state_data(self, robot_state: RobotState, sim_time: float):
        pass

    def add_feet_contact_data(self, robot_state: RobotState, sim_time: float):
        pass

    def add_policy_data(self, policy: Policy, sim_time: float):
        pass

    def plot(self, save_fig=True, save_data=True, log_dir=None):
        for plot_manager in self.plot_manager_list:
            plot_manager.plot(save_fig, save_data, log_dir)

    def close(self):
        for plot_manager in self.plot_manager_list:
            plot_manager.close()
