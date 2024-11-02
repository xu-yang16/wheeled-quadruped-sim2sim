import os.path as osp
import yaml, cv2
import numpy as np
import mujoco as mj
import mujoco.viewer as mj_viewer
from loguru import logger
from .yaml_mamager import save_to_yaml, load_from_yaml


class PassiveViewerWrapper:
    def __init__(
        self,
        model,
        data,
        yml_config="sim_config.yml",
        video_save_path="test.mp4",
        pause=False,
    ):
        self.pause = pause
        # load config
        self.config = load_from_yaml(
            filename=osp.join(osp.dirname(__file__), yml_config)
        )
        self.model, self.data = model, data
        self.viewer = mj_viewer.launch_passive(
            model,
            data,
            key_callback=lambda key: self.key_callback(key),
            show_left_ui=self.config["ui"]["show_left_ui"],
            show_right_ui=self.config["ui"]["show_right_ui"],
        )

        self.viewer.cam.azimuth = self.config["viewer"]["azimuth"]
        self.viewer.cam.elevation = self.config["viewer"]["elevation"]
        self.viewer.cam.lookat = np.array(self.config["viewer"]["lookat"])
        self.viewer.cam.distance = self.config["viewer"]["distance"]

        self.set_viewer_camera(body_name=self.config["viewer"]["track_body"])

        with self.viewer.lock():
            self.viewer.user_scn.flags[mj.mjtRndFlag.mjRND_REFLECTION] = self.config[
                "viewer"
            ]["relefction"]
            self.viewer.user_scn.flags[mj.mjtRndFlag.mjRND_SHADOW] = self.config[
                "viewer"
            ]["shadow"]
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = self.config[
                "viewer"
            ]["pertforce"]
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = self.config[
                "viewer"
            ]["contactpoint"]
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = self.config[
                "viewer"
            ]["contactforce"]

        # init render for video recording
        self.last_video_frame_time = 0
        self.model.vis.global_.offwidth = 1920
        self.model.vis.global_.offheight = 1080
        frame_width, frame_height, self.fps = 1920, 1080, 30
        self.renderer = mj.Renderer(self.model, height=frame_height, width=frame_width)
        self.output_mp4_path = video_save_path
        if self.output_mp4_path is not None:
            self.video_writer = cv2.VideoWriter(
                video_save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.renderer.width, self.renderer.height),
            )
        else:
            self.video_writer = None
        # Camera settings
        self.cam = mj.MjvCamera()
        mj.mjv_defaultCamera(self.cam)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.35, -130, -20
        self.cam.lookat[0], self.cam.lookat[1], self.cam.lookat[2] = 0.0, 0.0, 0.2

    def update(self):
        self.viewer.sync()
        self.record_video()

    def key_callback(self, keycode):
        if chr(keycode) == "C":
            self.dump_current_config()
            logger.info("Dumped current configuration")
        elif chr(keycode) == "P":
            self.pause = not self.pause
            logger.info(f"Pause: {self.pause}")

    def set_viewer_camera(self, body_name="trunk"):
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
        self.viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = body_id

    def record_video(self):
        if (
            self.output_mp4_path is None
            or self.viewer is None
            or self.video_writer is None
        ):
            return
        sim_time = self.data.time
        if sim_time - self.last_video_frame_time >= 1.0 / self.fps:
            # move camera to the pose of the viewer
            self.cam.azimuth = self.viewer.cam.azimuth
            self.cam.elevation = self.viewer.cam.elevation
            self.cam.distance = self.viewer.cam.distance
            self.cam.lookat = self.viewer.cam.lookat

            # render and record
            self.renderer.update_scene(self.data, self.cam)
            image = self.renderer.render()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # add timestamp text to the image
            timestamp_text = f"Time: {sim_time:.2f}"
            cv2.putText(
                image,
                timestamp_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            self.video_writer.write(image)

            # Update frame time and frame count
            self.last_video_frame_time = sim_time

    def dump_current_config(self):
        logger.warning(
            f"Camera position: {self.viewer.cam.lookat}, distance: {self.viewer.cam.distance}, elevation: {self.viewer.cam.elevation}, azimuth: {self.viewer.cam.azimuth}"
        )
        self.config["viewer"]["lookat"] = self.viewer.cam.lookat.tolist()
        self.config["viewer"]["distance"] = self.viewer.cam.distance
        self.config["viewer"]["elevation"] = self.viewer.cam.elevation
        self.config["viewer"]["azimuth"] = self.viewer.cam.azimuth
        save_to_yaml(self.config, filename="sim_config.yml")

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            logger.info(f"Video saved to {self.output_mp4_path}")
