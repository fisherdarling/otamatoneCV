from loc import Loc
from functools import total_ordering
from head import Head
import random
from scipy import stats
import numpy as np

import matplotlib.pyplot as plt


@total_ordering
class StaffLine(Loc):
    def from_ys(ys):
        center = sum(ys) / len(ys)

        return StaffLine(center)

    def center(self):
        return self.y

    def __init__(self, y):
        super()
        self.y = y

    def __eq__(self, other):
        return self.y == other.y

    def __ne__(self, other):
        return not (self == other)

    def __ge__(self, other):
        return self.y > other.y

    def __repr__(self):
        return f"StaffLine({self.y})"


class Staff:
    def __init__(self, y=None, lines=[]):
        self.y = y
        self.lines = []

    def lines(self):
        return self.lines

    def len(self):
        return len(self.lines)

    def center(self):
        return sum(map(lambda x: x.y, self.lines)) / len(self.lines)

    def add_line(self, line: StaffLine):
        self.lines.append(line)
        self.lines.sort()

    def get_note(self, head: Head):
        raise Exception("Not implemented")

    def __repr__(self):
        return f"Staff({self.center():.2f})"

    def extract_runs(frame, col):
        # the values in the column, col, from the current frame
        # e.g., [0,1,1,1,1,1,1,0,0,0,0,0,0,0,etc]
        data = frame[:, col]

        white_runs = []
        black_runs = []

        black_count, white_count = 0, 0

        for e in data:
            if e == 0:
                black_count += 1

                if white_count > 0:
                    white_runs.append(white_count)
                    white_count = 0
            elif e == 1:
                white_count += 1

                if black_count > 0:
                    black_runs.append(black_count)
                    black_count = 0

        if white_count > 0:
            white_runs.append(white_count)
        if black_count > 0:
            black_runs.append(black_count)

        return (black_runs, white_runs)

    # | 25% |----------| 25% |

    def extract_staff_metrics(frame):
        NUM_SELECTED_COL = 20
        random.seed()

        frame_width = frame.shape[1]
        half_frame_width = frame_width // 2
        start_col = half_frame_width - (half_frame_width // 2)
        possible_cols = range(start_col, start_col+half_frame_width)

        # if the image is so small that it doesn't allow us to grab a random NUM_SELECTED_COL
        # number of columns from the center 50%, then we will just not process this image
        if len(possible_cols) < NUM_SELECTED_COL:
            exit()

        random_col = random.sample(possible_cols, NUM_SELECTED_COL)

        black_modes = []
        white_modes = []

        for col in random_col:
            black_runs, white_runs = Staff.extract_runs(frame, col)
            black_modes += list(stats.mode(black_runs)[0])
            white_modes += list(stats.mode(white_runs)[0])

        # plt.hist(black_modes)
        # plt.hist(white_modes)
        # plt.show()
        # plt.waitforuserinput()

        staffline_height = int(np.median(black_modes))
        staffspace_height = int(np.median(white_modes))

        return (staffline_height, staffspace_height)


if __name__ == "__main__":
    print("Hello")
