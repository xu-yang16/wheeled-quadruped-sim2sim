import time
import os.path as osp
import yaml
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
from loguru import logger
from .yaml_mamager import save_to_yaml, load_from_yaml


class GlfwViewerWrapper:

    def __init__(self, model, data, yml_config="sim_config.yml"):
        # load config
        self.config = load_from_yaml(
            filename=osp.join(osp.dirname(__file__), yml_config)
        )
        self.model, self.data = model, data

        # for gui control
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_mouse_posx = 0
        self.last_mouse_posy = 0

        glfw.init()
        self.window = glfw.create_window(1000, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        opt = mj.MjvOption()  # visualization options
        opt.flags[mj.mjtRndFlag.mjRND_REFLECTION] = self.config["viewer"]["relefction"]
        opt.flags[mj.mjtRndFlag.mjRND_SHADOW] = self.config["viewer"]["shadow"]
        opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = self.config["viewer"][
            "pertforce"
        ]  # show perturbation force
        opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = self.config["viewer"][
            "contactpoint"
        ]  # show contact point
        opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = self.config["viewer"][
            "contactforce"
        ]  # show contact force
        self.opt = opt
        mj.mjv_defaultOption(self.opt)

        # # set camera configuration
        cam = mj.MjvCamera()  # Abstract camera
        cam.azimuth = self.config["viewer"]["azimuth"]
        cam.elevation = self.config["viewer"]["elevation"]
        cam.lookat = np.array(self.config["viewer"]["lookat"])
        cam.distance = self.config["viewer"]["distance"]
        self.cam = cam
        self.set_viewer_camera(body_name=self.config["viewer"]["track_body"])
        mj.mjv_defaultCamera(self.cam)

        self.scene = mj.MjvScene(self.model, maxgeom=200)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.mouse_scroll)

    def update(self):
        mj.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mj.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )

        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        mj.mjr_render(viewport, self.scene, self.context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(self.window)
        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

    def set_viewer_camera(self, body_name="trunk"):
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
        self.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = body_id

    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_R:
            pass
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

    def close(self):
        glfw.terminate()
