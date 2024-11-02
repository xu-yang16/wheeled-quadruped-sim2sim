import torch
import math
import copy

torch.pi = math.pi


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (
        q[:, qw] * q[:, qw]
        - q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        + q[:, qz] * q[:, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(torch.pi / 2.0, sinp), torch.asin(sinp)
    )

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (
        q[:, qw] * q[:, qw]
        + q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        - q[:, qz] * q[:, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


@torch.jit.script
def coordinate_rotation(axis: int, angle):
    s = torch.sin(angle)
    c = torch.cos(angle)

    R = torch.eye(3, dtype=torch.float, device=angle.device)
    R = R.reshape((1, 3, 3)).repeat(angle.size(0), 1, 1)

    if axis == 0:
        R[:, 1, 1] = c
        R[:, 2, 2] = c
        R[:, 1, 2] = s
        R[:, 2, 1] = -s
    elif axis == 1:
        R[:, 0, 0] = c
        R[:, 0, 2] = -s
        R[:, 2, 0] = s
        R[:, 2, 2] = c
    elif axis == 2:
        R[:, 0, 0] = c
        R[:, 0, 1] = s
        R[:, 1, 0] = -s
        R[:, 1, 1] = c

    return R


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


def torch_rand(lower, upper, num_rows, num_cols, device):
    return torch.rand(num_rows, num_cols, device=device) * (upper - lower) + lower
