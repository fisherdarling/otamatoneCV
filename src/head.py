from loc import Loc
from enum import Enum


class HeadType(Enum):
    FILLED = 1
    EMPTY = 2


class Head(Loc):
    def __init__(self, type=None, centroid=None, bounding_box=None):
        super().__init__(centroid, bounding_box)
        self.type = type
