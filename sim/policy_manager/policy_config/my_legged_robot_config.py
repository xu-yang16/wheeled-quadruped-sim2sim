from .base_config import BaseConfig


class MyLeggedRobotCfg(BaseConfig):
    class env:
        num_actor_observations = 83
        num_supervised_observations = 3
        num_vae_recovered_observations = 34
        num_observation_history = 6
        num_actions = 20

    class init_state:
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FR_hip_joint": -0.1,  # [rad]
            "FL_hip_joint": 0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_thigh_joint": 1.08,  # [rad]
            "FL_thigh_joint": 1.08,  # [rad]
            "RR_thigh_joint": 1.08,  # [rad]
            "RL_thigh_joint": 1.08,  # [rad]
            "FR_calf_joint": -2.16,  # [rad]
            "FL_calf_joint": -2.16,  # [rad]
            "RR_calf_joint": -2.16,  # [rad]
            "RL_calf_joint": -2.16,  # [rad]
            "FR_wheel_joint": 0.0,  # [rad]
            "FL_wheel_joint": 0.0,  # [rad]
            "RR_wheel_joint": 0.0,  # [rad]
            "RL_wheel_joint": 0.0,  # [rad]
        }

    class commands:
        num_commands = 3  # default: vel_x, yaw_rate, height
        resampling_time = 10.0  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-0.7, 0.7]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]

    class observation:
        clip_observations = 100.0

        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            height = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            wheel_dof_vel = 0.1

    class control:
        clip_actions = 3.0
        clip_wheel_actions = 40.0
        leg_stiffness = 40.0  # [N*m/rad]
        leg_damping = 1.0  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        wheel_action_scale = 5.0

    class pmtg:
        gait_type = "trot"  # hybrid: [trot, walk, driving]
        max_clearance = 0.1
        body_height = 0.2
        z_updown_height_func = ["cubic_up", "cubic_down"]
        max_horizontal_offset = 0.05

        train_mode = False

        delta_phi_scale = 0.2
        residual_angle_scale = 0.2
        residual_xyz_scale = 0.2

        max_delta_phi = 1.0
        max_residual_xyz = 0.3
