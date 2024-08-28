from furniture.furniture import Furniture
from furniture.square_table import SquareTable
from furniture.square_table_oneleg import SquareTableOneLeg
from furniture.table_leg import TableLeg


def furniture_factory(furniture: str) -> Furniture:
    if furniture == "square_table":
        return SquareTable()
    if furniture == "square_table_oneleg":
        return SquareTableOneLeg()
    if furniture == "table_leg":
        return TableLeg()
    # elif furniture == "desk":
    #     return Desk()
    # elif furniture == "round_table":
    #     return RoundTable()
    # elif furniture == "drawer":
    #     return Drawer()
    # elif furniture == "chair":
    #     return Chair()
    # elif furniture == "lamp":
    #     return Lamp()
    # elif furniture == "cabinet":
    #     return Cabinet()
    # elif furniture == "stool":
    #     return Stool()
    # elif furniture == "one_leg":
    #     return OneLeg()
    else:
        raise ValueError(f"Unknown furniture type: {furniture}")
