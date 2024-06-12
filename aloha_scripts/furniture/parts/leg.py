from furniture.parts.part import Part
from utils.pose import get_mat
import numpy as np

class Leg(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)

        self.rel_pose_from_center[self.tag_ids[0]] = get_mat(
            [0, 0, -self.tag_offset], [0, 0, 0]
        )
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [-self.tag_offset, 0, 0], [0, np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [0, 0, self.tag_offset], [0, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [self.tag_offset, 0, 0], [0, -np.pi / 2, 0]
        )