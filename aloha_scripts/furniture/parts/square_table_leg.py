from furniture.parts.leg import Leg

class SquareTableLeg(Leg):
    def __init__(self, part_config, part_idx):
        self.tag_offset = 0.015
        self.half_width = 0.015
        super().__init__(part_config, part_idx)
