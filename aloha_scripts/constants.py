### Task parameters
from typing import Dict, Any

from utils.pose import get_mat

DATA_DIR = '/media/hdd/group1_dataset'
TASK_CONFIGS = {
    'aloha_wear_shoe':{
        'dataset_dir': DATA_DIR + '/aloha_wear_shoe',
        'num_episodes': 50,
        'episode_len': 1000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_furniture':{
        'dataset_dir': DATA_DIR + '/aloha_furniture',
        'num_episodes': 50,
        'episode_len': 2000,
        'camera_names': ['cam_low', 'cam_left_wrist']
    },
}

config: Dict[str, Any] = {
    "furniture": {
        "detection_hz": 30,  # Check cam fps high enought
        "tag_family": "tag16h5",
        "base_tags": [0, 1, 2, 3],  # TODO what?
        "base_tag_size": 0.047,
        "rel_pose_from_coordinate": {  # TODO
            0: get_mat([-0.03, -0.03, 0], [0, 0, 0]),
            1: get_mat([0.03, -0.03, 0], [0, 0, 0]),
            2: get_mat([-0.03, 0.03, 0], [0, 0, 0]),
            3: get_mat([0.03, 0.03, 0], [0, 0, 0]),
        },
        "square_table": {
            "tag_size": 0.0195,
            "square_table_top": {
                "ids": [4, 5, 6, 7],
            },
            "square_table_leg1": {
                "ids": [8, 9, 10, 11],
            },
            "square_table_leg2": {
                "ids": [12, 13, 14, 15],
            },
            "square_table_leg3": {
                "ids": [16, 17, 18, 19],
            },
            "square_table_leg4": {
                "ids": [20, 21, 22, 23],
            },
        },
        "table_leg": {
            "tag_size": 0.0195,
            "square_table_leg1": {
                "ids": [8, 9, 10, 11],
            },
        },
    },
    "camera": {
        "cam_low": {
            # Camera parameters fx, fy, cx, cy
            "intr_param": [635.6, 637.4, 311.6, 231.5],
        },
        "cam_high": {
            # Camera parameters fx, fy, cx, cy
            "intr_param": [635.6, 637.4, 311.6, 231.5],
        },
        "cam_left_wrist": {
            # Camera parameters fx, fy, cx, cy
            "intr_param": [635.6, 637.4, 311.6, 231.5],
        },
        "cam_right_wrist": {
            # Camera parameters fx, fy, cx, cy
            "intr_param": [635.6, 637.4, 311.6, 231.5],
        },
    },
}

### ALOHA fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
# MASTER_GRIPPER_JOINT_CLOSE = -0.6842 # TODO: does this correspond to newer 3d printed parts? on teleop this doesnt fully close the puppet gripper
MASTER_GRIPPER_JOINT_CLOSE = -0.31753402948379517
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
