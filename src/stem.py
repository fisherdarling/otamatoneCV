from loc import Loc
from enum import Enum
import numpy as np
import math
from functools import total_ordering
import cv2

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

        self.c = line_center(self.a[0], self.a[1], self.b[0], self.b[1])
        self.aabb = [min(self.a[0], self.b[0]), min(self.a[1], self.b[1]), max(self.a[0], self.b[0]) - min(
            self.a[0], self.b[0]), max(self.a[1], self.b[1]) - min(self.a[1], self.b[1])]

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
        # print(f"Combine Similar {len(lines)}")

        # lines.sort()
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

    def x_center(self):
        return (self.a[0] + self.b[0]) / 2

    def bb(self):
        x_min = min(self.a[0], self.b[0])
        x_max = max(self.a[0], self.b[0])
        y_min = min(self.a[1], self.b[1])
        y_max = max(self.a[1], self.b[1])

        return x_min, x_max, y_min, y_max

    def draw(self, img):
        bb = self.bounding_box()
        cv2.rectangle(img, (bb[0], bb[1]), (bb[0] + bb[2],
                                            bb[1] + bb[3]), (0, 255, 0), thickness=2)

        if self.type:
            cv2.putText(img, self.type.name, (bb[0] + bb[2] + 5, bb[1] + bb[3] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=1)

    def intersects_rect(self, x, y, w, h):
        s1 = [(x, y), (x + w, y)]
        s2 = [(x, y), (x, y + h)]
        s3 = [(x + w, y), (x + w, y + h)]
        s4 = [(x, y + h), (x + w, y + h)]

        a = self.a
        b = self.b

        return intersects(a, b, s1[0], s1[1]) \
            or intersects(a, b, s2[0], s2[1]) \
            or intersects(a, b, s3[0], s3[1]) \
            or intersects(a, b, s4[0], s4[1])

    def __repr__(self):
        if self.type:
            return self.type.name
        else:
            return f"[{self.a} -> {self.b}]"

    def erode(self, delta, slice, note_aabb):
        delta = int(self.staffline_height * 1.4)
        n_x, _, _, n_h = note_aabb
        x_min, x_max, y_min, y_max = self.bb()

        # note_slice = self.binary[n_y:n_y + n_h, n_x:n_x + n_w]

        x_min -= delta
        x_max += delta
        y_min -= delta
        y_max += delta

        stem_w = x_max - x_min
        x_start = x_min - n_x

        slice = note_slice[0:n_h, x_start:x_start + stem_w]

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (x_max - x_min, 2))

        return cv2.erode(slice, kernel, iterations=2), x_start, stem_w


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


# From: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


# Return true if line segments AB and CD intersect
def intersects(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


if __name__ == "__main__":
    line = [5, 10, 5, 5]
    a = Stem(line=line)

    line = [4, 11, 4, 5]
    b = Stem(line=line)

    print(f"{a} {b}")

    l = [a, b]
    print(sorted(l))
