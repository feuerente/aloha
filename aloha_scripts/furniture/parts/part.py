from abc import ABC

from furniture.parts.pose_filter import PoseFilter


class Part(ABC):
    def __init__(self, part_config, part_idx: int):
        # Each camera has a filter.
        self.pose_filter = [PoseFilter(), PoseFilter()]
        self.part_idx = part_idx
        self.tag_ids = part_config["ids"]
        self.rel_pose_from_center = {}  # should be set in subclass.
