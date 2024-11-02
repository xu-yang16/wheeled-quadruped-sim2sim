import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import sys, time, threading
import os.path as osp

from scipy.spatial.transform import Rotation as R

# config
import torch

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from sim.policy_manager.policy_config.my_legged_robot_config import MyLeggedRobotCfg
from sim.policy_manager.PMTG import PMTrajectoryGenerator

from loguru import logger
import rerun as rr


def load_policy(logdir):
    import torch

    body = torch.jit.load(logdir + "/actor.jit")
    adaptation_module = torch.jit.load(logdir + "/adaptation_module.jit")

    def policy(actor_obs, obs_history):
        latent = adaptation_module.forward(
            obs_history.float()
            .reshape(
                -1,
            )
            .to("cpu")
            .unsqueeze(0)
        )
        action = body.forward(torch.cat((actor_obs.float().to("cpu"), latent), dim=-1))
        return action

    return policy


class State:
    def __init__(self):
        # imu
        self.imu_acc = np.zeros(3)
        self.imu_omega = np.zeros(3)
        self.imu_quat = np.array([1, 0, 0, 0.0])  # w, x, y, z
        # base_lin_vel
        self.lin_vel = np.zeros(3)
        # encoder
        self.q = np.zeros(12)
        self.dq = np.zeros(16)
        # torque applied
        self.tau = np.zeros(16)
        # contact force
        self.foot_contact_force = np.zeros((4, 3))

        # parameter for cheater mode
        self.base_position = np.zeros(3)
        self.base_quat = np.array([1, 0, 0, 0.0])  # w, x, y, z
        self.base_velocity = np.zeros(3)
        self.base_ang_velocity = np.zeros(3)

    @property
    def actual_vel(self):
        return np.array(
            [self.base_velocity[0], self.base_velocity[1], self.base_ang_velocity[2]]
        )

    @property
    def quat_xyzw_cheat(self):
        return np.array(
            [self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]]
        )

    @property
    def quat_xyzw(self):
        return np.array(
            [self.imu_quat[1], self.imu_quat[2], self.imu_quat[3], self.imu_quat[0]]
        )

    @property
    def obs_cheat(self):
        base_lin_vel = self.base_velocity
        base_ang_vel = self.base_ang_velocity
        projected_gravity = quatRotateInverse(
            self.quat_xyzw_cheat, np.array([0, 0, -1])
        )
        dof_pos = self.q
        dof_vel = self.dq
        return base_lin_vel, base_ang_vel, projected_gravity, dof_pos, dof_vel

    @property
    def obs(self):
        base_lin_vel = self.lin_vel
        base_ang_vel = self.imu_omega
        projected_gravity = quatRotateInverse(self.quat_xyzw, np.array([0, 0, -1]))
        dof_pos = self.q
        dof_vel = self.dq
        return base_lin_vel, base_ang_vel, projected_gravity, dof_pos, dof_vel


class Command:
    q_des_leg = np.zeros(12)
    dq_des_wheel = np.zeros(4)


class UserInputManger:
    def __init__(self):
        self.user_inputs_lib = [
            # {"duration": 5, "vel": [0.0, 0.0, 0.0]},
            # {"duration": 5, "vel": [0.0, 0.0, 0.0]},
            {"duration": 5, "vel": [-2.8, 0, 0.0]},
            {"duration": 5, "vel": [0.0, 1.5, 0.0]},
            {"duration": 5, "vel": [0.0, 0.0, 1.5]},
            {"duration": 5, "vel": [0.0, 0.0, 0.0]},
            # {"duration": 5, "vel": [0.0, 0.0, 1.0]},
            # {"duration": 5, "vel": [0.0, 0.5, 0.0]},
            # {"duration": 5, "vel": [1.0, 0.0, 0.2]},
            # {"duration": 5, "vel": [1.0, 0.0, 0.3]},
        ]
        self.user_inputs_lib = [
            # {"duration": 5, "vel": [0.0, 0.0, 0.0]},
            # {"duration": 5, "vel": [0.0, 0.0, 0.0]},
            {"duration": 5, "vel": [1.0, 0, 0.0]},
            {"duration": 5, "vel": [0.0, 0.8, 0.0]},
            {"duration": 5, "vel": [0.0, 0.0, 0.8]},
            {"duration": 5, "vel": [0.0, 0.0, 0.0]},
            # {"duration": 5, "vel": [0.0, 0.0, 1.0]},
            # {"duration": 5, "vel": [0.0, 0.5, 0.0]},
            # {"duration": 5, "vel": [1.0, 0.0, 0.2]},
            # {"duration": 5, "vel": [1.0, 0.0, 0.3]},
        ]
        self.user_input_start_time = 0.0
        self.elapsed_time = 0.0
        self.user_inputs = np.zeros(3)
        self.current_input_idx = 0

    def reset(self, current_time):
        self.user_input_start_time = current_time
        self.elapsed_time = 0.0
        self.user_inputs = np.zeros(3)
        self.current_input_idx = 0

    def get_user_input(self, current_time, transition_ratio=0.2):
        self.elapsed_time = current_time - self.user_input_start_time
        duration = self.user_inputs_lib[self.current_input_idx]["duration"]
        user_inputs = np.array(self.user_inputs_lib[self.current_input_idx]["vel"])
        if self.elapsed_time >= duration:
            self.current_input_idx += 1
            if self.current_input_idx > len(self.user_inputs_lib) - 1:
                self.current_input_idx = 0
            self.user_input_start_time = current_time
            self.elapsed_time = 0.0

        if self.elapsed_time <= transition_ratio * duration:
            ratio = self.elapsed_time / (transition_ratio * duration)
            return self.interpolate(np.zeros(3), user_inputs, ratio)
        elif self.elapsed_time <= (1 - transition_ratio) * duration:
            return user_inputs
        elif self.elapsed_time <= duration:
            ratio = (self.elapsed_time - (1 - transition_ratio) * duration) / (
                transition_ratio * duration
            )
            return self.interpolate(user_inputs, np.zeros(3), ratio)

    def interpolate(self, start_cmd, end_cmd, ratio):
        """
        start_cmd: numpy array
        end_cmd: numpy array
        ratio: float, [0.0, 1.0]
        """
        assert 0.0 <= ratio <= 1.0
        interpolated_cmd = (1 - ratio) * start_cmd + ratio * end_cmd
        return interpolated_cmd


class MujocoSimulator:
    def __init__(self, model_file) -> None:
        self.xml_path = model_file
        hip_ab = 0.1
        hip_ad = 1.08
        knee = -2.16
        self.default_joint_pos = (
            [-hip_ab, hip_ad, knee, 0]
            + [hip_ab, hip_ad, knee, 0]
            + [-hip_ab, hip_ad, knee, 0]
            + [hip_ab, hip_ad, knee, 0]
        )
        self.state = State()
        self.num_joints = 16
        self.print_camera_config = 1
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_mouse_posx = 0
        self.last_mouse_posy = 0

        # plot buffer
        self.timestamp = np.empty(0)
        self.user_input_buffer = np.empty((0, 3))
        self.state_buffer = np.empty((0, 3))

        # use rerun to plot

    def initController(self, policy):
        # user input manager
        self.user_cmds = np.zeros(3)
        self.user_input_manager = UserInputManger()
        self.user_input_manager.reset(self._get_time())
        # controller
        self.cfg = MyLeggedRobotCfg
        self.obs = np.zeros(self.cfg.env.num_actor_observations)
        self.obs_history = np.zeros(
            (self.cfg.env.num_observation_history, self.cfg.env.num_actor_observations)
        )
        self.actions = np.zeros(16)
        self.last_actions = np.zeros(16)
        self.command = Command()
        self.pmtg = PMTrajectoryGenerator(
            device="cpu",
            num_envs=1,
            param=MyLeggedRobotCfg.pmtg,
        )
        self.pmtg.reset(index_list=[0], current_time=self._get_time())
        self.controller = policy

        # default pos
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
        self.default_dof_pos = np.array(
            [self.cfg.init_state.default_joint_angles[name] for name in joint_names]
        )

        self.controller_thread = threading.Thread(target=self.run)
        self.controller_thread.start()

    def resetController(self):
        self.pmtg.reset(index_list=[0], current_time=self._get_time())
        self.actions = np.zeros(16)
        self.last_actions = np.zeros(16)
        self.obs = np.zeros(self.cfg.env.num_actor_observations)
        self.obs_history = np.zeros(
            (self.cfg.env.num_observation_history, self.cfg.env.num_actor_observations)
        )

    def _get_time(self):
        return self.data.time

    def run(self):
        self.updateState()
        self.obs, self.obs_history = self.getObs()
        leg_actions, wheel_actions = self.step(self.obs, self.obs_history)
        current_time = self._get_time()
        # while not rospy.is_shutdown():
        while not glfw.window_should_close(self.window):
            self.low_level_controller(leg_actions, wheel_actions)
            mj.mj_step(self.model, self.data)
            time.sleep(0.001)
            # object_index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "trunk")
            # self.data.xfrc_applied[object_index] = [40.0, 0.0, 0, 0.0, 0.0, 0.0]
            # FIXME:
            # self.data.qpos[0:3] = np.array([0, 0, 0.3])
            # self.data.qpos[3:7] = np.array([1, 0, 0, 0])
            # rr.log(f"Forward_Vel/actual", rr.Scalar(self.state.base_velocity[0]))
            # rr.log(f"Forward_Vel/desired", rr.Scalar(self.user_cmds[0]))
            # rr.log(f"Lateral_Vel/actual", rr.Scalar(self.state.base_velocity[1]))
            # rr.log(f"Lateral_Vel/desired", rr.Scalar(self.user_cmds[1]))
            # rr.log(f"Angular_Vel/actual", rr.Scalar(self.state.base_ang_velocity[2]))
            # rr.log(f"Angular_Vel/desired", rr.Scalar(self.user_cmds[2]))

            if self._get_time() - current_time < 0.01:
                pass
            else:
                current_time = self._get_time()
                self.updateState()
                self.obs, self.obs_history = self.getObs()

                leg_actions, wheel_actions = self.step(self.obs, self.obs_history)

            # save buffer for plot
            self.timestamp = np.append(self.timestamp, self._get_time())
            self.user_input_buffer = np.append(
                self.user_input_buffer, self.user_cmds.reshape((1, -1)), axis=0
            )
            self.state_buffer = np.append(
                self.state_buffer,
                self.state.actual_vel.reshape((1, 3)),
                axis=0,
            )

    def resetSim(self):
        mj.mj_resetData(self.model, self.data)
        # reset qpos
        if self.data.qpos.size > 12:  # float base model
            self.data.qpos[7:] = self.default_joint_pos.copy()
        else:  # fixed base model
            self.data.qpos = self.default_joint_pos.copy()
        mj.mj_forward(self.model, self.data)
        mj.mj_step(self.model, self.data)

    def initSimulator(
        self, show_perturbation=True, show_contact_point=True, show_contact_force=True
    ):
        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(self.xml_path)  # MuJoCo model
        self.data = mj.MjData(self.model)  # MuJoCo data
        self.opt = mj.MjvOption()  # visualization options
        self.opt.flags[mj.mjtVisFlag.mjVIS_COM] = 1
        self.opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = (
            show_perturbation  # show perturbation force
        )
        self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = (
            show_contact_point  # show contact point
        )
        self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = (
            show_contact_force  # show contact force
        )
        mj.mjv_defaultOption(self.opt)
        # set camera configuration
        self.cam = mj.MjvCamera()  # Abstract camera
        self.cam.azimuth = 90
        self.cam.elevation = -45
        self.cam.distance = 2
        self.cam.lookat = np.array([0.0, 0.0, 0])
        mj.mjv_defaultCamera(self.cam)
        # robot init state
        if self.data.qpos.size > 12:  # float base model
            self.data.qpos[7:] = self.default_joint_pos.copy()
        else:  # fixed base model
            self.data.qpos = self.default_joint_pos.copy()
        mj.mj_forward(self.model, self.data)
        # Init GLFW library, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(1000, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        # initialize visualization data structures
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        # print camera configuration (help to initialize the view)
        if self.print_camera_config == 1:
            logger.info(
                f"cam.azimuth = {self.cam.azimuth}, cam.elevation = {self.cam.elevation}, cam.distance = {self.cam.distance}"
            )
            logger.info(
                f"cam.lookat =np.array([{self.cam.lookat[0]}, {self.cam.lookat[1]}, {self.cam.lookat[2]}])"
            )
        for i in range(16):
            # if i % 4 == 3:
            #     self.setVelocityServo(i, 10.0)
            # else:
            self.setTorqueServo(i)
        # self.setVelocityServo(4, 10)
        # self.setTorqueServo(7)
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.mouse_scroll)

    def low_level_controller(self, leg_actions, wheel_actions):
        cmd = Command()
        cmd.q_des_abad = leg_actions[[0, 3, 6, 9]]
        cmd.q_des_hip = leg_actions[[1, 4, 7, 10]]
        cmd.q_des_knee = leg_actions[[2, 5, 8, 11]]
        cmd.dq_des_wheel = wheel_actions

        kp = self.cfg.control.leg_stiffness
        kd = self.cfg.control.leg_damping
        for i in range(4):
            self.data.ctrl[4 * i] = kp * (
                cmd.q_des_abad[i] - self.data.qpos[7 + 4 * i]
            ) + kd * (0.0 - self.data.qvel[6 + 4 * i])
            self.data.ctrl[4 * i + 1] = kp * (
                cmd.q_des_hip[i] - self.data.qpos[8 + 4 * i]
            ) + kd * (0.0 - self.data.qvel[7 + 4 * i])
            self.data.ctrl[4 * i + 2] = kp * (
                cmd.q_des_knee[i] - self.data.qpos[9 + 4 * i]
            ) + kd * (0.0 - self.data.qvel[8 + 4 * i])
            self.data.ctrl[4 * i + 3] = 2.0 * (
                cmd.dq_des_wheel[i] - self.data.qvel[9 + 4 * i]
            )

    def runSimulation(self):
        # self.ros_thread.start()
        # key and mouse control
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.mouse_scroll)
        while not glfw.window_should_close(self.window):
            # mj.mj_forward(self.model, self.data)
            time_prev = time.time()
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
            # Update scene and render
            self.cam.lookat = np.array(self.data.qpos[0:3])
            self.opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = 1  # show perturbation force
            self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = 1  # show contact point
            self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = 1  # show contact force
            mj.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                self.cam,
                mj.mjtCatBit.mjCAT_ALL.value,
                self.scene,
            )
            mj.mjr_render(viewport, self.scene, self.context)
            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)
            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()
            # while (self.data.time-time_prev < 1.0/60.0):
            #     print(self.data.time-time_prev)
            logger.info(f"duration={time.time()- time_prev}")
        glfw.terminate()

    def updateState(self):
        # imu
        self.state.imu_acc = self.data.sensor("Body_Acc").data.astype(
            np.double
        )  # include gravity 9.81
        self.state.imu_omega = self.data.sensor("Body_Gyro").data.astype(np.double)
        self.state.imu_quat = self.data.sensor("Body_Quat").data.astype(
            np.double
        )  # w, x, y, z
        # frame_lin_vel
        self.state.lin_vel = self.data.sensor("Body_Vel").data.astype(np.double)
        # contact info
        for i, leg in enumerate(["FR", "FL", "RR", "RL"]):
            self.state.foot_contact_force[i, :] = self.data.sensor(
                f"{leg}_contact_force_sensor"
            ).data.astype(np.double)

        # encoder
        i = 0
        for leg in ["FR", "FL", "RR", "RL"]:
            for joint in ["hip", "thigh", "calf"]:
                self.state.q[i] = (self.data.sensor(f"{leg}_{joint}_pos")).data.astype(
                    np.double
                )
                i += 1
        i = 0
        for leg in ["FR", "FL", "RR", "RL"]:
            for joint in ["hip", "thigh", "calf", "wheel"]:
                self.state.dq[i] = (self.data.sensor(f"{leg}_{joint}_vel")).data.astype(
                    np.double
                )
                self.state.tau[i] = (
                    self.data.sensor(f"{leg}_{joint}_torque")
                ).data.astype(np.double)
                i += 1

        # parameter for cheater mode
        self.state.base_position = self.data.qpos[0:3]
        self.state.base_quat = self.data.qpos[3:7]
        self.state.base_velocity = self.data.qvel[0:3]
        self.state.base_ang_velocity = self.data.qvel[3:6]

    def step(self, obs, obs_history):
        policy_outputs = self.controller(
            torch.from_numpy(obs).unsqueeze(0),
            torch.from_numpy(obs_history).unsqueeze(0),
        )
        policy_outputs = policy_outputs
        delta_phi = policy_outputs[:, 0:4] * self.cfg.pmtg.delta_phi_scale
        delta_phi = torch.clip(
            delta_phi, -self.cfg.pmtg.max_delta_phi, self.cfg.pmtg.max_delta_phi
        )
        residual_xyz = policy_outputs[:, 4:16] * self.cfg.pmtg.residual_xyz_scale
        residual_xyz = torch.clip(
            residual_xyz,
            -self.cfg.pmtg.max_residual_xyz,
            self.cfg.pmtg.max_residual_xyz,
        )
        residual_angle = torch.zeros(1, 12)
        # self.data.qpos[3:7]: w, x, y, z
        base_quat = torch.from_numpy(self.state.quat_xyzw).unsqueeze(0)  # x, y, z, w
        pmtg_joints = self.pmtg.get_action(
            delta_phi, residual_xyz, residual_angle, base_quat, self._get_time()
        )
        wheel_vel = policy_outputs[:, 16:20] * self.cfg.control.wheel_action_scale

        clip_actions = self.cfg.control.clip_actions
        clip_wheel_actions = self.cfg.control.clip_wheel_actions
        leg_actions = torch.clip(pmtg_joints, -clip_actions, clip_actions)
        wheel_actions = torch.clip(wheel_vel, -clip_wheel_actions, clip_wheel_actions)

        self.last_actions[:] = self.actions[:]
        self.actions = (
            torch.cat([leg_actions, wheel_actions], dim=1).squeeze(0).detach().numpy()
        )

        return (
            leg_actions.squeeze(0).detach().numpy(),
            wheel_actions.squeeze(0).detach().numpy(),
        )

    def getObs(self, cheat_mode=True):
        # self.data.qpos[3:7]: w, x, y, z
        if cheat_mode:
            base_lin_vel, base_ang_vel, projected_gravity, dof_pos, dof_vel = (
                self.state.obs_cheat
            )
        else:
            base_lin_vel, base_ang_vel, projected_gravity, dof_pos, dof_vel = (
                self.state.obs
            )

        # user input from manager
        self.user_cmds = self.user_input_manager.get_user_input(self._get_time())
        cpg_phase_info = self.pmtg.update_observation().detach().numpy()

        obs_scales = self.cfg.observation.obs_scales
        commands_scale = np.array(
            [obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel]
        )
        dof_vel = dof_vel * obs_scales.dof_vel
        dof_vel[[3, 7, 11, 15]] *= obs_scales.wheel_dof_vel
        self.obs = np.concatenate(
            (
                base_ang_vel * obs_scales.ang_vel,  # 3
                projected_gravity,  # 3
                (dof_pos - self.default_dof_pos) * obs_scales.dof_pos,  # 12
                dof_vel,  # 16
                cpg_phase_info,  # 14
                self.actions,  # 16
                self.last_actions,  # 16
                self.user_cmds * commands_scale,  # 3
            ),
            axis=None,
        )
        if np.all(self.obs_history == 0):
            self.obs_history = np.tile(
                self.obs, (self.cfg.env.num_observation_history, 1)
            )
        else:
            self.obs_history = np.roll(self.obs_history, -1, axis=0)
            self.obs_history[-1] = self.obs
        return self.obs, self.obs_history

    # set motor mode
    def setPostionServo(self, actuator_no, kp):
        """set motor's control mode to be position mode"""
        self.model.actuator_gainprm[actuator_no, 0:3] = [kp, 0, 0]
        self.model.actuator_biasprm[actuator_no, 0:3] = [0, -kp, 0]
        self.model.actuator_biastype[actuator_no] = 1
        # print(self.model.actuator_biastype)

    def setVelocityServo(self, actuator_no, kv):
        """set motor's control mode to be velocity mode"""
        self.model.actuator_gainprm[actuator_no, 0:3] = [kv, 0, 0]
        self.model.actuator_biasprm[actuator_no, 0:3] = [0, 0, -kv]
        self.model.actuator_biastype[actuator_no] = 1
        # print(self.model.actuator_biastype)

    def setTorqueServo(self, actuator_no):
        """set motor's control mode to be torque mode"""
        self.model.actuator_gainprm[actuator_no, 0:3] = [1, 0, 0]
        self.model.actuator_biasprm[actuator_no, 0:3] = [0, 0, 0]
        self.model.actuator_biastype[actuator_no] = 0
        # print(self.model.actuator_biastype)

    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_R:
            self.resetSim()
            self.resetController()
        if act == glfw.PRESS and key == glfw.KEY_S:
            print("Pressed key s")
        if act == glfw.PRESS and key == glfw.KEY_UP:
            pass
        if act == glfw.PRESS and key == glfw.KEY_DOWN:
            pass
        if act == glfw.PRESS and key == glfw.KEY_LEFT:
            pass
        if act == glfw.PRESS and key == glfw.KEY_RIGHT:
            pass

    # update button state
    def mouse_button(self, window, button, act, mods):
        self.button_left = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self.button_middle = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )
        self.button_right = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )
        glfw.get_cursor_pos(window)  # update mouse position

    def mouse_scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, 0.05 * yoffset, self.scene, self.cam)

    def mouse_move(self, window, xpos, ypos):
        # compute mouse displacement, save
        dx = xpos - self.last_mouse_posx
        dy = ypos - self.last_mouse_posy
        self.last_mouse_posx = xpos
        self.last_mouse_posy = ypos
        # # determine action based on mouse button
        if self.button_right:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
        elif self.button_middle:
            action = mj.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        width, height = glfw.get_window_size(window)  # get current window size
        mj.mjv_moveCamera(
            self.model, action, dx / height, dy / height, self.scene, self.cam
        )

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 9))
        logger.info(f"shape={self.timestamp.shape}")
        logger.info(f"user input shape={self.user_input_buffer.shape}")
        logger.info(f"state shape={self.state_buffer.shape}")
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(
                self.timestamp, self.user_input_buffer[:, i], label="Desired Velocity"
            )
            plt.plot(self.timestamp, self.state_buffer[:, i], label="Actual Velocity")
            plt.xlabel("Time/s")
            plt.ylabel("Velocity")
            plt.title("Velocity")
            plt.legend()
        plt.tight_layout()
        plt.show()


# utils
def quatRotateInverse(q: np.array, v: np.array):
    """q = [w, z, y, z]"""
    q_w = q[3]
    q_vec = np.array(q[0:3])
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * 2.0 * q_w
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


if __name__ == "__main__":
    model_xml = "resources/wheeled_go1/xml/scene_terrain.xml"
    sim = MujocoSimulator(model_xml)
    sim.initSimulator()
    # load controller
    policy = load_policy(
        osp.join(osp.dirname(__file__), "../sim/policy_manager", "runs/PMTG_flat/Dec06")
    )
    policy = load_policy(
        osp.join(
            osp.dirname(__file__),
            "../sim/policy_manager",
            "runs/PMTG_flat/Dec09_23-25-01_a1e57a2",
        )
    )
    sim.initController(policy)

    rr.init("wheeled_go1_sim", spawn=True)
    rr.connect()
    sim.runSimulation()

    sim.plot()

    rr.disconnect()
