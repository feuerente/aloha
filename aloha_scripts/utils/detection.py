import time
from collections import deque
from multiprocessing import shared_memory

import numpy as np
from numpy.linalg import inv

import utils.transform as T
from constants import config
from robot_utils import ImageRecorder, read_detect
from utils.apriltag import AprilTag
from utils.frequency import set_frequency
from utils.pose import comp_avg_pose


def detection_loop(parts, num_parts, tag_size, lock, shm, shared_dict):
    print("Start detection")

    april_tag = AprilTag(tag_size)

    image_recorder = ImageRecorder(init_node=True)
    # image_recorder = shared_dict["image_recorder"]
    cam_low_frame = image_recorder.get_cam_low_image()
    cam_low_intr_param = config["camera"]["cam_low"]["intr_param"]
    cam_low_to_base = get_cam_to_base(img=cam_low_frame, cam_intr=cam_low_intr_param, april_tag=april_tag)
    cam_high_frame = image_recorder.get_cam_high_image()
    cam_high_intr_param = config["camera"]["cam_high"]["intr_param"]
    cam_high_to_base = get_cam_to_base(img=cam_high_frame, cam_intr=cam_high_intr_param, april_tag=april_tag)

    detection_deltas = deque(maxlen=5)

    while True:
        detection_start = time.time()
        detection = _get_parts_poses(
            parts,
            num_parts,
            april_tag,
            cam_low_to_base,
            cam_high_to_base,
            image_recorder,
        )
        detection_end = time.time()
        detection_delta = detection_end - detection_start
        detection_deltas.append(detection_delta)
        detection_delta_avg = sum(detection_deltas) / len(detection_deltas) if detection_deltas else 0
        deteciton_frequency = 1 / detection_delta_avg
        # print(f"Detection Frequency: {deteciton_frequency}")

        parts_poses_shm = shared_memory.SharedMemory(name=shm[0])
        parts_founds_shm = shared_memory.SharedMemory(name=shm[1])

        parts_poses = np.ndarray(
            shape=(num_parts * 7,), dtype=np.float32, buffer=parts_poses_shm.buf
        )
        parts_found = np.ndarray(
            shape=(num_parts,), dtype=bool, buffer=parts_founds_shm.buf
        )

        lock.acquire()
        parts_poses[:] = detection[0]
        parts_found[:] = detection[1]
        lock.release()


@set_frequency(config["furniture"]["detection_hz"])
def _get_parts_poses(
        parts,
        num_parts,
        april_tag,
        cam_low_to_base,
        cam_high_to_base,
        image_recorder: ImageRecorder,
):
    max_fail = 1
    part_idx = 0
    fail_count = 0
    parts_poses = np.zeros(
        (num_parts * 7,), dtype=np.float32
    )  # 3d positional, 4d rotational (quaternion).
    parts_founds = []

    (
        tags_low,
        tags_high,
    ) = read_detect(april_tag, image_recorder)

    # Debug pose
    # print(f"tags_low:  {tags_low}")
    # print(f"tags_high: {tags_high}")
    print(f"tags_low:  {[tag for tag in tags_low]}")
    # print(f"tags_high: {[tag for tag in tags_high]}")

    for part in parts:
        part_idx = part.part_idx

        cam_low_pose = _get_parts_pose(part, tags_low)
        cam_high_pose = _get_parts_pose(part, tags_high)
        if cam_low_pose is not None:
            cam_low_pose = cam_low_to_base @ cam_low_pose
        if cam_high_pose is not None:
            cam_high_pose = cam_high_to_base @ cam_high_pose

        if cam_low_pose is not None or cam_high_pose is not None:
            pose_low = (
                part.pose_filter[0].filter(cam_low_pose) if cam_low_pose is not None else None
            )
            pose_high = (
                part.pose_filter[1].filter(cam_high_pose) if cam_high_pose is not None else None
            )
            pose = comp_avg_pose([pose_low, pose_high])

            # print(f"pose: {T.mat2pose(pose)[0]}") # Debug pose

            parts_poses[part_idx * 7: (part_idx + 1) * 7] = np.concatenate(
                list(T.mat2pose(pose))
            ).astype(np.float32)
            part_idx += 1
            parts_founds.append(True)
        else:
            # Detection failed.
            fail_count += 1
            if fail_count >= max_fail:
                part_idx += 1
                parts_founds.append(False)
            else:
                time.sleep(0.01)
                # Read camera and detect tags again.
                (
                    tags_low,
                    tags_high,
                ) = read_detect(april_tag, image_recorder)
        cam_low_pose = inv(cam_low_to_base) @ cam_low_pose if cam_low_pose is not None else None
        cam_high_pose = inv(cam_high_to_base) @ cam_high_pose if cam_high_pose is not None else None
        # TODO is there something missing here?

    parts_founds = np.array(parts_founds, dtype=bool)
    return (
        parts_poses,
        parts_founds,
    )


def _get_parts_pose(part, tags):
    poses = []
    for tag_id in part.tag_ids:
        tag = tags.get(tag_id)
        if tag is None:
            continue
        # Get 3d position of 3 points.
        pose = T.to_homogeneous(tag.pose_t, tag.pose_R)
        # Compute where the anchor tag is.
        pose = pose @ np.linalg.inv(part.rel_pose_from_center[tag.tag_id])

        poses.append(pose)
    if len(poses) == 0:
        return None
    return comp_avg_pose(poses)


def get_cam_to_base(cam=None, img=None, cam_intr=None, april_tag=None):
    """Get homogeneous transforms that maps camera points to base points."""
    if april_tag is None:
        april_tag = AprilTag(config["furniture"]["base_tag_size"])
    if img is None:
        color_frame, _ = cam.get_frame()
    else:
        color_frame = img

    if cam_intr is not None:
        intr = cam_intr
    else:
        intr = cam.intr_param

    tags = april_tag.detect_id(color_frame, intr)

    trials = 10
    cam_to_bases = []
    for _ in range(trials):
        for base in config["furniture"]["base_tags"]:
            base_tag = tags.get(base)
            if base_tag is not None:
                rel_pose = config["furniture"]["rel_pose_from_coordinate"][base]
                transform = T.to_homogeneous(
                    base_tag.pose_t, base_tag.pose_R
                ) @ np.linalg.inv(rel_pose)
                cam_to_bases.append(np.linalg.inv(transform))

        tags = april_tag.detect_id(color_frame, intr)  # TODO new trial but same frame
        time.sleep(0.01)

    #  All the base tags are not detected.
    if len(cam_to_bases) == 0:
        return None
        # raise Exception(f"camera {cam_num}: Base tags are not detected.")
    cam_to_base = comp_avg_pose(cam_to_bases)
    return cam_to_base
