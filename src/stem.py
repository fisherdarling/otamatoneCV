from loc import Loc
from enum import Enum
import numpy as np
import math
from functools import total_ordering


from scipy.spatial.distance import directed_hausdorff


class StemType(Enum):
    NORMAL = 1
    EIGHTH_FLAG = 2
    SIXTEENTH_FLAG = 3
    EIGHTH_BEAM = 4
    SIXTEENTH_BEAM = 5


@total_ordering
class Stem(Loc):
    def __init__(self, type=None, line=None, a=None, b=None, centroid=None, bounding_box=None):
        super().__init__(centroid, bounding_box)

        self.a = a
        self.b = b

        if line is not None:
            a = (line[0], line[1])
            b = (line[2], line[3])

            if a[1] > b[1]:
                self.a = a
                self.b = b
            else:
                self.a = b
                self.b = a

            # p = sorted([a, b])

            # self.a = p[0]
            # self.b = p[1]

        self.line = line
        self.type = type

    def __eq__(self, other):
        self.a == other.a and self.b == other.b

    def __ne__(self, other):
        return not (self == other)

    def __ge__(self, other):
        return self.a > other.a or self.b > other.b

    def __le__(self, other):
        return self.a < other.a or self.b < other.b

    def __hash__(self):
        return hash((hash(self.a), hash(self.b)))

    def combine_similar_lines(lines, staffspace_height):
        print(f"Combine Similar {len(lines)}")

        lines.sort()
        lines.reverse()

        combined = []

        start = lines.pop()
        while len(lines) > 0:
            top = lines.pop()

            check_a_dist = dist(start.a, top.a)
            check_b_dist = dist(start.b, top.b)

            # print(start, top, check_a_dist, check_b_dist)

            if top.a[0] > 1916 and top.a[0] < 1960 \
                    and top.a[1] > 1090 and top.a[1] < 1150:
                print("WEIRD", start, top, check_a_dist, check_b_dist)

            if check_a_dist < staffspace_height and check_b_dist < staffspace_height:
                start = Stem.combine(start, top)
                # print("combined")
            else:
                # print("starting new")
                combined.append(start)
                start = top

        combined.append(start)
        # for i in range(len(lines) - 1):
        #     check_a_dist = dist(lines[i].a, lines[i + 1].a)
        #     check_b_dist = dist(lines[i].b, lines[i + 1].b)

        #     print(f"{check_a_dist} || {check_b_dist}")

        # min_line = None
        # a_dist = 100000.0
        # b_dist = 100000.0

        # for j in range(i, len(lines)):
        #     if i == j or lines[j] in used:
        #         print(f"{lines[j]} already used")

        #         continue

        #     check_a_dist = dist(lines[i].a, lines[j].a)
        #     check_b_dist = dist(lines[i].b, lines[j].b)

        #     if check_a_dist > staffspace_height or check_b_dist > staffspace_height:
        #         continue

        #     if check_a_dist < a_dist and check_b_dist < b_dist:  # and s < 10.0:
        #         min_line = j
        #         a_dist = check_a_dist
        #         b_distt = check_b_dist

        # if min_line is not None:
        #     new_line = Stem.combine(lines[i], lines[min_line])
        #     combined.append(new_line)
        #     # used.add(i)
        #     # used.add(j)
        #     used.add(lines[min_line])
        # else:
        #     combined.append(lines[i])
        #     # used.add(i)

        return combined

    def combine(stem1, stem2):
        new_a = p_avg(stem1.a, stem2.a)
        new_b = p_avg(stem1.b, stem2.b)

        return Stem(a=new_a, b=new_b)

    def y_min(self):
        return min(self.a[1], self.b[1])

    def y_max(self):
        return max(self.a[1], self.b[1])

    def bb(self):
        x_min = min(self.a[0], self.b[0])
        x_max = max(self.a[0], self.b[0])
        y_min = min(self.a[1], self.b[1])
        y_max = max(self.a[1], self.b[1])

        return x_min, x_max, y_min, y_max

    def __repr__(self):
        return f"[{self.a} -> {self.b}]"


# def dist(a, b):
#     return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def dist(a, b):
    return abs(a[0] - b[0])


def p_avg(a, b):
    return int(round((a[0] + b[0]) / 2)), int(round((a[1] + b[1]) / 2))


def line_angle(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1 + 0.1)
    return abs(np.arctan(slope))


def line_center(x1, y1, x2, y2):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return [center_x, center_y]

# def combine_lines(line1, line2):


# def similarity(s1, s2):
#     u = np.array([s1.a, s1.b])
#     v = np.array([s2.a, s2.b])

#     return directed_hausdorff(u, v)[0]


if __name__ == "__main__":
    line = [5, 10, 5, 5]
    a = Stem(line=line)

    line = [4, 11, 4, 5]
    b = Stem(line=line)

    print(f"{a} {b}")

    l = [a, b]
    print(sorted(l))
