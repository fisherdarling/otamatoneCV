from loc import Loc
from enum import Enum
import numpy as np


class StemType(Enum):
    NORMAL = 1
    EIGHTH_FLAG = 2
    SIXTEENTH_FLAG = 3
    EIGHTH_BEAM = 4
    SIXTEENTH_BEAM = 5


class Stem(Loc):
    def __init__(self, type=None, line=None, centroid=None, bounding_box=None):
        super().__init__(centroid, bounding_box)
        self.line = line
        self.type = type


def line_angle(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1 + 0.1)
    return abs(np.arctan(slope))


def line_center(x1, y1, x2, y2):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return [center_x, center_y]
