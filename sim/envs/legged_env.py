import numpy as np
from scipy.spatial.transform import Rotation
import mujoco as mj
from collections import OrderedDict

from .mj_sim import MjSim
from ..common.utils import MotorCommand

# for rendering
from .utils.visual_utils import render_vector, render_sphere

from loguru import logger


class JointInfo:
    def __init__(
        self,
        joint_name: str,
        actuator_name: str,
        joint_id: int,
        actuator_id: int,
    ):
        self.joint_name = joint_name
        self.actuator_name = actuator_name
        self.joint_id = joint_id
        self.actuator_id = actuator_id

        # state related
        self.joint_pos = None
        self.joint_vel = None
        self.joint_torque = None

        # control related
        self.joint_control_type = None  # torque, pd, p, d
        self.ctrl = None

    def copmute_joint_torque(
        self, control_type: str, ctrl: float, kp: float = 40.0, kd: float = 1.0
    ):
        self.joint_control_type = control_type
        self.ctrl = ctrl

        if type == "pd":
            target_pos = ctrl
            target_vel = 0
            return kp * (target_pos - self.joint_pos) + kd * (
                target_vel - self.joint_vel
            )
        elif type == "p":
            target_pos = ctrl
            return kp * (target_pos - self.joint_pos)
        elif type == "d":
            target_vel = ctrl
            return kd * (target_vel - self.joint_vel)
        elif type == "torque":
            return ctrl
        else:
            raise ValueError(
                f"Invalid control type: {type} for joint {self.joint_name}"
            )

    def update_joint_state(self, pos: float, vel: float, torque: float):
        self.joint_pos = pos
        self.joint_vel = vel
        self.joint_torque = torque


class LeggedEnv(MjSim):
    def __init__(
        self,
        model_xml: str,
        simulate_dt: float,
        sim_step_per_control: int = 1,
        legs_joint_names: OrderedDict = None,
        feet_geom_names: list = None,
        viewer_type="passive",  # glfw or passive
        video_save_path=None,
    ):
        self.sim_time_per_control = sim_step_per_control
        super().__init__(
            model_xml,
            simulate_dt,
            viewer_type,
            video_save_path,
        )

        assert (
            legs_joint_names is not None
        ), "Please provide the joint names associated with each of the legs."
        self.legs_order = legs_joint_names.keys()

        # feet related
        self.feet_geom_names = feet_geom_names
        self.feet_geom_ids, self.feet_body_ids = self.get_feet_geom_body_id(
            feet_geom_names
        )
        logger.info(f"Feet geometry IDs: {self.feet_geom_ids}, {self.feet_body_ids}")

        # extract all actuator info
        self.all_dof_names = []
        for idx in range(self.model.njnt):
            jnt_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, idx)
            if jnt_name == None:
                continue
            self.all_dof_names.append(jnt_name)
        logger.info(f"DOF names: {self.dof_names}")
        self.joint_info_dict = OrderedDict()
        for actuator_idx in range(self.model.nu):
            act_name = self.model.actuator(actuator_idx).name
            mj_actuator_id = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, act_name
            )
            # Get the joint index associated with the actuator
            joint_id = self.model.actuator_trnid[mj_actuator_id, 0]
            # Get the joint name from the joint index
            joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            self.joint_info_dict[joint_name] = JointInfo(
                joint_name, act_name, joint_id, mj_actuator_id
            )

    def reset(self, default_joint_pos: np.array, height=0.3):
        mj.mj_resetData(self.model, self.data)
        # reset floating base model
        self.data.qpos[2] = height
        self.data.qpos[7:] = default_joint_pos.copy()
        mj.mj_forward(self.model, self.data)  # forward kinematics

        if self._viewer_wrapper is not None:
            self._viewer_wrapper.update()

    def step(self, motor_cmds: MotorCommand):
        for _ in range(self.sim_time_per_control):
            # compute torques
            torques = np.zeros(self.model.nu)
            for idx, jnt_name in enumerate(motor_cmds.joint_names):
                if jnt_name not in self.joint_info_dict:
                    logger.warning(
                        f"Joint {jnt_name} not found in the model. Skipping..."
                    )
                    continue

                joint_id = self.joint_info_dict[jnt_name].joint_id
                actuator_id = self.joint_info_dict[jnt_name].actuator_id

                motor_pos = self.data.qpos[6 + joint_id]
                motor_vel = self.data.qvel[5 + joint_id]
                kp = motor_cmds.Kps[idx]
                kd = motor_cmds.Kds[idx]
                torques[actuator_id] = kp * (
                    motor_cmds.target_pos[idx] - motor_pos
                ) + kd * (motor_cmds.target_vel[idx] - motor_vel)

            # forward simulation
            self.simulate(torques)

    # for domain rand
    def _set_ground_friction(
        self,
        tangential_coeff: float = 1.0,  # Default MJ tangential coefficient
        torsional_coeff: float = 0.005,  # Default MJ torsional coefficient
        rolling_coeff: float = 0.0,  # Default MJ rolling coefficient
    ):
        """Initialize ground friction coefficients using a specified distribution."""
        pass
        for geom_id in range(self.mjModel.ngeom):
            geom_name = mj.mj_id2name(self.mjModel, mj.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and geom_name.lower() in [
                "ground",
                "floor",
                "hfield",
                "terrain",
            ]:
                self.mjModel.geom_friction[geom_id, :] = [
                    tangential_coeff,
                    torsional_coeff,
                    rolling_coeff,
                ]
                # print(f"Setting friction for {geom_name} to: {tangential_coeff, torsional_coeff, rolling_coeff}")
            elif (
                geom_id in self._feet_geom_id
            ):  # Set the same friction coefficients for the feet geometries
                self.mjModel.geom_friction[geom_id, :] = [
                    tangential_coeff,
                    torsional_coeff,
                    rolling_coeff,
                ]
            else:
                pass

    def _perturb_body(self, body_name="trunk", perturb_force=np.zeros(6)):
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
        self.data.xfrc_applied[body_id] = perturb_force

    # world pos and ori
    @property
    def world_pos(self):
        return self.data.qpos[:3]

    @property
    def world_quat_wxyz(self):
        return self.data.qpos[3:7]  # w, x, y, z

    @property
    def world_quat_xyzw(self):
        return np.roll(self.data.qpos[3:7], -1)

    @property
    def world_base_rot_matrix(self):
        quat_wxyz = self.data.qpos[3:7]  # w, x, y, z
        quat_xyzw = np.roll(quat_wxyz, -1)
        return Rotation.from_quat(quat_xyzw).as_matrix()

    @property
    def world_base_ori_euler_xyz(self):
        """Returns the base orientation in Euler XYZ angles (roll, pitch, yaw) in the world reference frame."""
        quat_wxyz = self.data.qpos[3:7]  # w, x, y, z
        quat_xyzw = np.roll(quat_wxyz, -1)  # x, y, z, w
        return Rotation.from_quat(quat_xyzw).as_euler("xyz")

    @property
    def projected_gravity(self):
        R = self.world_base_rot_matrix
        return R.T @ np.array([0, 0, -1.0])

    # world vel and acc
    @property
    def world_lin_vel(self):
        return self.data.qvel[:3]

    @property
    def world_ang_vel(self):
        return self.data.qvel[3:6]

    @property
    def world_lin_acc(self):
        return self.data.qacc[:3]

    @property
    def world_ang_acc(self):
        return self.data.qacc[3:6]

    # base vel and acc
    @property
    def base_lin_vel(self):
        R = self.world_base_rot_matrix
        return R.T @ self.world_lin_vel

    @property
    def base_ang_vel(self):
        R = self.world_base_rot_matrix
        return R.T @ self.world_ang_vel

    @property
    def base_lin_acc(self):
        R = self.world_base_rot_matrix
        return R.T @ self.world_lin_acc

    @property
    def base_ang_acc(self):
        R = self.world_base_rot_matrix
        return R.T @ self.world_ang_acc

    @property
    def base_ang_acc(self):
        R = self.world_base_rot_matrix
        return R.T @ self.world_ang_acc

    # dof pos and vel
    @property
    def dof_names(self):
        return self.all_dof_names

    @property
    def dof_pos(self):
        return self.data.qpos[7:]

    @property
    def dof_vel(self):
        return self.data.qvel[6:]

    @property
    def dof_torque(self):
        return self.data.actuator_force

    # feet pos and vel
    @property
    def world_feet_pos(self):
        """Calculate the position of the feet in the world frame."""
        feet_pos_in_base_frame = np.zeros((len(self.feet_geom_ids), 3))
        for idx in range(len(self.feet_body_ids)):
            body_id = self.feet_body_ids[idx]
            pos = self.data.xpos[body_id]
            feet_pos_in_base_frame[idx] = pos
        return feet_pos_in_base_frame

    @property
    def base_feet_pos(self):
        R = self.world_base_rot_matrix
        return R.T @ (self.world_feet_pos - self.world_pos)

    @property
    def base_feet_vel(self):
        feet_vel = np.zeros((len(self.feet_geom_ids), 3))
        for idx in range(len(self.feet_body_ids)):
            body_id = self.feet_body_ids[idx]
            jacp = np.zeros((3, self.model.nv))
            mj.mj_jacBody(self.model, self.data, jacp=jacp, jacr=None, body=body_id)
            jacp = jacp.reshape((3, -1))
            feet_vel[idx] = jacp @ self.data.qvel.copy()
        return feet_vel

    @property
    def feet_contact_state(self):
        return np.linalg.norm(self.world_feet_contact_forces, axis=1) > 0

    @property
    def world_feet_contact_forces(self):
        contact_forces = np.zeros((len(self.feet_geom_ids), 3))
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            logger.error(f"Contact: {contact.geom1}, {contact.geom2}")
            if contact.geom2 in self.feet_geom_ids:
                result = np.zeros(6)
                mj.mj_contactForce(self.model, self.data, i, result)
                grf_glo = np.zeros(3)
                mj.mju_mulMatTVec(grf_glo, contact.frame.reshape((3, 3)), result[:3])
                contact_forces[self.feet_geom_ids.index(contact.geom2)] = grf_glo
        return contact_forces

    @property
    def world_feet_contact_pos(self):
        contact_pos = np.zeros((len(self.feet_geom_ids), 3))
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom2 in self.feet_geom_ids:
                contact_pos[self.feet_geom_ids.index(contact.geom2)] = contact.pos
        return contact_pos

    @property
    def com(self):
        """Calculate the center of mass (CoM) of the entire robot in world frame."""
        total_mass = 0.0
        com = np.zeros(3)
        for i in range(self.model.nbody):
            body_mass = self.model.body_mass[i]
            body_com = self.data.subtree_com[i]
            com += body_mass * body_com
            total_mass += body_mass
        com /= total_mass
        return com

    # from sensors
    @property
    def base_lin_vel_sensor(self):
        # from sensor
        return self.data.sensor("Body_Vel").data.astype(np.double)

    @property
    def base_ang_vel_sensor(self):
        # from sensor
        return self.data.sensor("Body_Gyro").data.astype(np.double)

    @property
    def base_lin_acc_sensor(self):
        self.data.sensor("Body_Acc").data.astype(np.double)  # include gravity 9.81

    # utils
    def get_feet_geom_body_id(self, feet_geom_names: list):
        feet_geom_ids = []
        feet_body_ids = []
        for foot_geom_name in feet_geom_names:
            foot_geom_id = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_GEOM, foot_geom_name
            )
            if foot_geom_id == -1:
                logger.warning(
                    f"Foot geometry {foot_geom_name} not found in the model."
                )
            feet_geom_ids.append(foot_geom_id)
            body_id = self.model.geom_bodyid[foot_geom_id]
            feet_body_ids.append(body_id)
        return feet_geom_ids, feet_body_ids

    @property
    def feet_jacobians(self):
        return

    @property
    def base_inertia(self) -> np.ndarray:
        # Initialize the full mass matrix
        mass_matrix = np.zeros((self.model.nv, self.model.nv))
        mj.mj_fullM(self.model, mass_matrix, self.data.qM)

        # Extract the 3x3 rotational inertia matrix of the base (assuming the base has 6 DoFs)
        return mass_matrix[3:6, 3:6]
