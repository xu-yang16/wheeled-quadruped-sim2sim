from __future__ import annotations

import mujoco as mj
import numpy as np
from mujoco.viewer import Handle
from scipy.spatial.transform import Rotation


def cross2(
    a: np.ndarray, b: np.ndarray
) -> np.ndarray:  # See https://github.com/microsoft/pylance-release/issues/3277
    return np.cross(a, b)


def render_vector(
    viewer: Handle,
    vector: np.ndarray,
    pos: np.ndarray,
    scale: float,
    diameter: float = 0.02,
    color: np.ndarray = np.array([1, 0, 0, 1]),
    geom_id: int = -1,
) -> int:
    """
    Function to render a vector in the Mujoco viewer.

    Args:
        viewer (Handle): The Mujoco viewer.
        vector (np.ndarray): The vector to render.
        pos (np.ndarray): The position of the base of vector.
        scale (float): The scale of the vector.
        color (np.ndarray): The color of the vector.
        geom_id (int, optional): The id of the geometry. Defaults to -1.
    Returns:
        int: The id of the geometry.
    """
    if geom_id < 0:
        # Instantiate a new geometry
        geom = mj.MjvGeom()
        geom.type = mj.mjtGeom.mjGEOM_ARROW
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1

    geom = viewer.user_scn.geoms[geom_id]

    # Define the a rotation matrix with the Z axis aligned with the vector direction
    vec_z = vector.squeeze() / np.linalg.norm(vector + 1e-5)
    # Define any orthogonal to z vector as the X axis using the Gram-Schmidt process
    rand_vec = np.random.rand(3)
    vec_x = rand_vec - (np.dot(rand_vec, vec_z) * vec_z)
    vec_x = vec_x / np.linalg.norm(vec_x)
    # Define the Y axis as the cross product of X and Z
    vec_y = cross2(vec_z, vec_x)

    ori_mat = Rotation.from_matrix(np.array([vec_x, vec_y, vec_z]).T).as_matrix()
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_ARROW,
        size=np.asarray([diameter, diameter, scale]),
        pos=pos,
        mat=ori_mat.flatten(),
        rgba=color,
    )
    geom.category = mj.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1

    return geom_id


def render_sphere(
    viewer: Handle,
    position: np.ndarray,
    diameter: float,
    color: np.ndarray,
    geom_id: int = -1,
) -> int:
    """
    Function to render a sphere in the Mujoco viewer.

    Args:
        viewer (Handle): The Mujoco viewer.
        position (np.ndarray): The position of the sphere.
        diameter (float): The diameter of the sphere.
        color (np.ndarray): The color of the sphere.
        geom_id (int, optional): The id of the geometry. Defaults to -1.
    Returns:
        int: The id of the geometry.
    """
    if geom_id < 0:
        # Instantiate a new geometry
        geom = mj.MjvGeom()
        geom.type = mj.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1

    geom = viewer.user_scn.geoms[geom_id]

    # Initialize the geometry
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=np.asarray([diameter / 2] * 3),  # Radius is half the diameter
        mat=np.eye(3).flatten(),
        pos=position,
        rgba=color,
    )

    geom.category = mj.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1

    return geom_id


def render_line(viewer: Handle, initial_point, target_point, width, color, geom_id=-1):
    """
    Function to render a line in the Mujoco viewer.

    Args:
        viewer (Handle): The Mujoco viewer.
        initial_point (np.ndarray): The initial point of the line.
        target_point (np.ndarray): The target point of the line.
        width (float): The width of the line.
        color (np.ndarray): The color of the line.
        geom_id (int, optional): The id of the geometry. Defaults to -1.
    Returns:
        int: The id of the geometry.
    """
    if geom_id < 0:
        # Instantiate a new geometry
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1

    geom = viewer.user_scn.geoms[geom_id]

    # Define the rotation matrix with the Z axis aligned with the line direction
    vector = target_point - initial_point
    length = np.linalg.norm(vector)
    if length == 0:
        return geom_id

    vec_z = vector / length

    # Use Gram-Schmidt process to find an orthogonal vector for X axis
    rand_vec = np.random.rand(3)
    vec_x = rand_vec - np.dot(rand_vec, vec_z) * vec_z
    vec_x /= np.linalg.norm(vec_x)

    # Define the Y axis as the cross product of X and Z
    vec_y = cross2(vec_z, vec_x)

    ori_mat = Rotation.from_matrix(np.array([vec_x, vec_y, vec_z]).T).as_matrix()

    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=np.array([width, length / 2 + width / 4, width]),
        pos=(initial_point + target_point) / 2,
        mat=ori_mat.flatten(),
        rgba=color,
    )

    return geom_id
