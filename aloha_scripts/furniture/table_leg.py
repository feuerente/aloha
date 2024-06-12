from constants import config
from furniture.furniture import Furniture
from furniture.parts.square_table_leg import SquareTableLeg
from furniture.parts.square_table_top import SquareTableTop


class TableLeg(Furniture):
    def __init__(self):
        super().__init__()
        furniture_conf = config["furniture"]["table_leg"]
        self.furniture_conf = furniture_conf

        self.tag_size = furniture_conf["tag_size"]

        self.parts = [
            SquareTableLeg(furniture_conf["square_table_leg1"], 0),
        ]
        self.num_parts = len(self.parts)
