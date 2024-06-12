from furniture.parts.part import Part


class TableTop(Part):
    def __init__(self, part_config: dict, part_idx: int):
        super().__init__(part_config, part_idx)
