from typing import List, Union

import numpy as np
import numpy.typing as npt

import utils.transform as T
from utils.averageQuaternions import averageQuaternions


def comp_avg_pose(poses):
    np.set_printoptions(suppress=True)
    quats = []
    positions = []
    for pose in poses:
        if pose is None:
            continue
        quats.append(T.convert_quat(T.mat2quat(pose[:3, :3]), "wxyz"))
        positions.append(pose[:3, 3])

    quats = np.stack(quats, axis=0)
    positions = np.stack(positions, axis=0)

    # print(f"avg position: {[pos for pos in positions]}")
    # Debug pose
    # print(f"comp_avg_pose position_error: {position_error}")

    avg_quat = averageQuaternions(quats).astype(np.float32)

    avg_rot = T.quat2mat(T.convert_quat(avg_quat, "xyzw"))
    avg_pos = positions.mean(axis=0)
    return T.to_homogeneous(avg_pos, avg_rot)


def get_mat(pos: List[float], angles: Union[List[float], npt.NDArray[np.float32]]):
    """Get homogeneous matrix given position and rotation angles.
    Args:
        pos: relative positions (x, y, z).
        angles: relative rotations (x, y, z) or 3x3 matrix.
    """
    transform = np.zeros((4, 4), dtype=np.float32)
    if not isinstance(angles, np.ndarray) or not len(angles.shape) == 2:
        transform[:3, :3] = np.eye(3) if not np.any(angles) else rot_mat(angles)
    else:
        if len(angles[0, :]) == 4:
            transform[:4, :4] = angles
        else:
            transform[:3, :3] = angles
    transform[3, 3] = 1.0
    transform[:3, 3] = pos
    return transform


def rot_mat(angles, hom: bool = False):
    """Given @angles (x, y, z), compute rotation matrix
    Args:
        angles: (x, y, z) rotation angles.
        hom: whether to return a homogeneous matrix.
    """
    x, y, z = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    if hom:
        M = np.zeros((4, 4), dtype=np.float32)
        M[:3, :3] = R
        M[3, 3] = 1.0
        return M
    return R


def is_similar_pose(pose1, pose2, ori_bound=0.99, pos_threshold=[0.01, 0.007, 0.007]):
    """Check if two poses are similar."""
    similar_rot = is_similar_rot(pose1[:3, :3], pose2[:3, :3], ori_bound)

    similar_pos = is_similar_pos(pose1[:3, 3], pose2[:3, 3], pos_threshold)

    return similar_rot and similar_pos


def is_similar_rot(rot1, rot2, ori_bound=0.99):
    if cosine_sim(rot1[:, 0], rot2[:, 0]) < ori_bound:
        return False
    if cosine_sim(rot1[:, 1], rot2[:, 1]) < ori_bound:
        return False
    if cosine_sim(rot1[:, 2], rot2[:, 2]) < ori_bound:
        return False
    return True


def is_similar_pos(pos1, pos2, pos_threshold=[0.01, 0.007, 0.007]):
    if np.abs(pos1[0] - pos2[0]) > pos_threshold[0]:  # x
        return False
    if np.abs(pos1[1] - pos2[1]) > pos_threshold[1]:  # y
        return False
    if len(pos1) > 2 and np.abs(pos1[2] - pos2[2]) > pos_threshold[2]:  # z
        return False
    return True


def cosine_sim(w, v):
    return np.dot(w, v) / (np.linalg.norm(w) * np.linalg.norm(v))
