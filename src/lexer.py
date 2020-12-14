import matplotlib.pyplot as plt
import numpy as np
import cv2

from head import HeadType


TONES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
NUM_TONES = 7


class Lexer:
    def __init__(self, binary, staffs, notes, staffspace_height):
        self.binary = binary
        self.staffs = staffs
        self.notes = notes
        self.staffspace_height = staffspace_height

        # self.generate_statistics()

    def generate_statistics(self):
        # print("here")
        pass
        # areas = list(map(lambda note: note.head.covered_area(), self.notes))
        # areas = list(map(lambda note: note.head.area, self.notes))
        # self.median_area_percent = np.median(areas)

        # print(self.median_area_percent)

        # plt.ioff()
        # plt.hist(areas)
        # plt.show()
        # plt.show(block=True)
        # cv2.waitKey()
        # plt.waitforbuttonpress()

    def classify_heads(self):
        to_delete = []

        for i, note in enumerate(self.notes):

            x, y, w, h = note.head.bounding_box()
            slice = self.binary[y:y+h, x + w // 4:x + (3 * w) // 4]
            n_rows, n_cols = slice.shape

            projection = np.count_nonzero(slice, axis=0)
            sum = np.sum(projection)
            area = h * (w // 2)

            if sum / area < 0.65:
                note.head.type = HeadType.EMPTY
            else:
                note.head.type = HeadType.FILLED

            if note.stem is None and note.head.type == HeadType.FILLED:
                to_delete.append(i)
                continue

            staff = self.nearest_staff(note)

            # Calculate the note head center:
            c_x, c_y = x + (w / 2), y + (h / 2)

            check_y = (c_y - staff.center()) / (self.staffspace_height / 2)
            idx = round(check_y)

            if abs(idx) > 16:
                to_delete.append(i)
                continue

            note.tone = TONES[idx % NUM_TONES]
            # print(c_x, c_y, round(check_y), staff)

        # to_delete.sort()
        # print(to_delete)

        while len(to_delete) > 0:
            self.notes.pop(to_delete.pop())

    def nearest_staff(self, note):
        _, c_y = note.centroid()

        return min(self.staffs, key=lambda staff: abs(staff.center() - c_y))

    def avg_staff_diff(self):
        total = 0.0

        for i in range(len(self.staffs) - 1):
            # print(self.staffs[i])

            total += (self.staffs[i + 1].center() - self.staffs[i].center())

        return total / (len(self.staffs) - 1)


def in_tolerance(value, target, tolerance):
    percent = abs(value - target) / target

    return percent < tolerance
