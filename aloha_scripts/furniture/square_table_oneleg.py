from constants import config
from furniture.furniture import Furniture
from furniture.parts.square_table_leg import SquareTableLeg
from furniture.parts.square_table_top import SquareTableTop


class SquareTableOneLeg(Furniture):
    def __init__(self):
        super().__init__()
        furniture_conf = config["furniture"]["square_table"]
        self.furniture_conf = furniture_conf

        self.tag_size = furniture_conf["tag_size"]

        self.parts = [
            SquareTableTop(furniture_conf["square_table_top"], 0),
            SquareTableLeg(furniture_conf["square_table_leg1"], 1),
        ]
        self.num_parts = len(self.parts)

        # TODO include rest
