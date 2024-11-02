import numpy as np


class MotorCommand:
    def __init__(
        self,
        joint_names: list,
        target_pos: np.array,
        target_vel: np.array,
        Kps: np.array,
        Kds: np.array,
    ):
        self.joint_names = joint_names
        self.target_pos = target_pos
        self.target_vel = target_vel
        self.Kps = Kps
        self.Kds = Kds

    def __str__(self):
        return f"MotorCommand: {self.joint_names}, {self.target_pos}, {self.target_vel}, {self.Kps}, {self.Kds}"
