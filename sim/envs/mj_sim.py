import time, threading
import numpy as np
import mujoco as mj

# for viewer
from .viewer.glfw_viewer import GlfwViewerWrapper
from .viewer.passive_viewer import PassiveViewerWrapper

# for logging
from loguru import logger


class MjSim:
    def __init__(
        self,
        model_xml,
        simulate_dt,
        viewer_type="passive",  # glfw or passive
        video_save_path=None,
    ):
        self.model = mj.MjModel.from_xml_path(model_xml)
        self.data = mj.MjData(self.model)

        self.pause = False

        # set simulation time step
        self.model.opt.timestep = simulate_dt

        if viewer_type == "glfw":
            self._viewer_wrapper = GlfwViewerWrapper(
                self.model,
                self.data,
                video_save_path=video_save_path,
                pause=self.pause,
            )
        elif viewer_type == "passive":
            self._viewer_wrapper = PassiveViewerWrapper(
                self.model,
                self.data,
                video_save_path=video_save_path,
                pause=self.pause,
            )
        else:
            logger.error(f"Invalid viewer type: {viewer_type}")
            self._viewer_wrapper = None

    @property
    def estimated_per_sim_in_real_time(self, repeat=100):
        start_time = time.time()
        for i in range(repeat):
            self.simulate(np.zeros(self.num_actuators))
        end_time = time.time()
        return (end_time - start_time) / repeat

    def simulate(self, torques):
        """
        Simulate the environment for one step.
        """
        if self.pause:
            pass
        else:
            self.data.ctrl[:] = torques[:]
            mj.mj_step(self.model, self.data)
        if self._viewer_wrapper is not None:
            self._viewer_wrapper.update()
            self.pause = self._viewer_wrapper.pause

    # utils
    def get_qpos_qvel(self, joint_name):
        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        pos_index = self.model.jnt_qposadr[joint_id]
        vel_index = self.model.jnt_dofadr[joint_id]
        return self.data.qpos[pos_index], self.data.qvel[vel_index]

    def set_actuator_value(self, act_name, value):
        actuator_id = self.actuator_names.index(act_name)
        self.data.ctrl[actuator_id] = value

    def close(self):
        if self._viewer_wrapper is not None:
            self._viewer_wrapper.close()

    @property
    def sim_time(self):
        return self.data.time

    @property
    def sim_dt(self):
        return self.model.opt.timestep

    @property
    def num_actuators(self):
        return len(self.model.actuator_actnum)

    @property
    def actuator_names(self):
        return [self.model.actuator(j).name for j in range(self.num_actuators)]
