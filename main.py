import os, time
import numpy as np
from loguru import logger

# load simulation manager
from sim.envs.legged_env import LeggedEnv

# load state manager
from sim.state_manager.state import RobotState

# load policy manager
from sim.policy_manager.policy import Policy

# load command manager
from sim.command_manager.automaic_command_generator import AutomaticCommandGenerator

# load plot manager
from sim.custom_plot_manager.custom_plot import PltManager
from sim.custom_plot_manager.rr_plot import RRManager
from sim.custom_plot_manager.frequent_used_plots import LeggedPlotManager

# for rendering
from sim.envs.utils.visual_utils import render_vector, render_sphere


class WheeledSim:
    def __init__(
        self,
        model_xml: str,
        policy_model_path: str,
        sim_dt: float = 0.002,
        control_dt: float = 0.02,
        use_plot="rr",  # or "custom"
    ):
        self.control_dt = control_dt
        decimation = int(control_dt / sim_dt)
        self.mj_sim = LeggedEnv(
            model_xml,
            simulate_dt=sim_dt,
            sim_step_per_control=decimation,
            legs_joint_names={
                "FR": [
                    "FR_hip_joint",
                    "FR_thigh_joint",
                    "FR_calf_joint",
                    "FR_wheel_joint",
                ],
                "FL": [
                    "FL_hip_joint",
                    "FL_thigh_joint",
                    "FL_calf_joint",
                    "FL_wheel_joint",
                ],
                "RR": [
                    "RR_hip_joint",
                    "RR_thigh_joint",
                    "RR_calf_joint",
                    "RR_wheel_joint",
                ],
                "RL": [
                    "RL_hip_joint",
                    "RL_thigh_joint",
                    "RL_calf_joint",
                    "RL_wheel_joint",
                ],
            },
            feet_geom_names=["FR_wheel", "FL_wheel", "RR_wheel", "RL_wheel"],
            viewer_type="passive",
            video_save_path=None,
        )

        self.command_manager = AutomaticCommandGenerator(loop=False)
        self.robot_state = RobotState(self.mj_sim, self.command_manager)
        self.policy = Policy(
            robot_state=self.robot_state, model_path=policy_model_path, device="cpu"
        )

        self.my_plot = LeggedPlotManager()
        if "rr" in use_plot:
            self.my_plot.add_plot_manager(RRManager())
        if "custom" in use_plot:
            self.my_plot.add_plot_manager(PltManager())

    @property
    def default_joint_pos(self):
        hip_ab = 0.1
        hip_ad = 1.08
        knee = -2.16
        default_joint_pos = np.array(
            [-hip_ab, hip_ad, knee, 0]
            + [hip_ab, hip_ad, knee, 0]
            + [-hip_ab, hip_ad, knee, 0]
            + [hip_ab, hip_ad, knee, 0]
        )
        return default_joint_pos

    def reset(self):
        sim_time = self.mj_sim.sim_time

        self.estimated_time_per_sim = 0.01  # self.mj_sim.estimated_per_sim_in_real_time
        logger.info(f"Estimated time per sim: {self.estimated_time_per_sim:.4f} sec")

        self.mj_sim.reset(default_joint_pos=self.default_joint_pos, height=0.3)

        self.command_manager.reset(sim_time)
        self.robot_state.reset()
        self.policy.reset()
        self.my_plot.reset()

    def run(self, total_time):
        start_time = time.time()

        while self.mj_sim.sim_time < total_time:
            sim_time = self.mj_sim.sim_time
            # update user input
            self.command_manager.update(sim_time, transition_ratio=0.2)

            # update policy
            motor_cmds = self.policy.update()

            # step simulation
            self.mj_sim.step(motor_cmds)
            # time.sleep(max(0, self.estimated_time_per_sim - (time.time() - start_time)))

            # rendering
            # self.debug_render()

            # for plot and data logging
            self.my_plot.update(self.robot_state, motor_cmds, self.policy, sim_time)

        self.my_plot.plot(save_fig=True, save_data=True, log_dir="logs")
        # close
        self.close()

    def debug_render(
        self,
        vis_com=True,
        vis_command=True,
        vis_contact=False,
        vis_feet=False,
    ):
        if vis_com:
            if not hasattr(self, "com_geom_id"):
                self.com_geom_id = -1
                self.com_projection_geom_id = -1
            self.com_geom_id = render_sphere(
                self.mj_sim._viewer_wrapper.viewer,
                self.mj_sim.com,
                0.1,
                np.array([1, 0, 0, 1]),
                self.com_geom_id,
            )
            com_projection = np.copy(self.mj_sim.com)
            com_projection[2] = 0
            self.com_projection_geom_id = render_sphere(
                self.mj_sim._viewer_wrapper.viewer,
                com_projection,
                0.1,
                np.array([1, 0, 0, 1]),
                self.com_projection_geom_id,
            )
        if vis_command:
            if not hasattr(self, "desired_vel_geom_id"):
                self.desired_vel_geom_id = -1
            vector_pos = np.copy(self.mj_sim.com)
            vector_pos[2] += 0.3
            desired_vel = self.robot_state.get_state("current_commands")
            desired_vel = np.array([desired_vel[0], desired_vel[1], 0])
            robot_base_rot = self.mj_sim.world_base_rot_matrix
            desired_vel = np.dot(robot_base_rot, desired_vel)
            vec_scale = np.linalg.norm(desired_vel)

            self.desired_vel_geom_id = render_vector(
                self.mj_sim._viewer_wrapper.viewer,
                vector=desired_vel,
                pos=vector_pos,
                scale=vec_scale,
                diameter=0.02,
                color=np.array([1, 0.5, 0, 0.7]),
                geom_id=self.desired_vel_geom_id,
            )
        if vis_contact:
            # render feet contact forces
            if not hasattr(self, "feet_contact_geom_ids"):
                self.feet_contact_geom_ids = [-1, -1, -1, -1]
            contact_forces = self.mj_sim.world_feet_contact_forces
            contact_pos = self.mj_sim.world_feet_contact_pos
            for idx in range(contact_forces.shape[0]):
                self.feet_contact_geom_ids[idx] = render_vector(
                    self.mj_sim._viewer_wrapper.viewer,
                    vector=contact_forces[idx, :],
                    pos=contact_pos[idx, :],
                    scale=0.002 * np.linalg.norm(contact_forces[idx, :]),
                    diameter=0.01,
                    color=np.array([0, 1, 0, 0.7]),
                    geom_id=self.feet_contact_geom_ids[idx],
                )
        if vis_feet:
            # render feet pos
            feet_pos = self.mj_sim.world_feet_pos
            feet_vel = self.mj_sim.base_feet_vel
            if not hasattr(self, "feet_geom_ids"):
                self.feet_geom_ids = [-1, -1, -1, -1]
                self.feet_vel_geom_ids = [-1, -1, -1, -1]
            for idx in range(feet_pos.shape[0]):
                self.feet_geom_ids[idx] = render_sphere(
                    self.mj_sim._viewer_wrapper.viewer,
                    feet_pos[idx, :],
                    0.06,
                    np.array([1, 0, 0, 1]),
                    self.feet_geom_ids[idx],
                )
                self.feet_vel_geom_ids[idx] = render_vector(
                    self.mj_sim._viewer_wrapper.viewer,
                    vector=feet_vel[idx, :],
                    pos=feet_pos[idx, :],
                    scale=0.5 * np.linalg.norm(feet_vel[idx, :]),
                    diameter=0.01,
                    color=np.array([0, 1, 1, 0.7]),
                    geom_id=self.feet_vel_geom_ids[idx],
                )

    def close(self):
        self.mj_sim.close()
        self.my_plot.close()


if __name__ == "__main__":
    my_wheeled_sim = WheeledSim(
        model_xml="resources/wheeled_go1/xml/scene_terrain.xml",
        policy_model_path="runs/PMTG_flat/Dec06",
        sim_dt=0.002,
        control_dt=0.01,
        use_plot=["custom"],
    )  # FIXME: turn this into yaml config, maybe hydra?
    my_wheeled_sim.reset()
    my_wheeled_sim.run(total_time=20.0)
