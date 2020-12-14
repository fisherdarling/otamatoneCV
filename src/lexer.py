import matplotlib.pyplot as plt
import numpy as np
import cv2

from head import HeadType


class Lexer:
    def __init__(self, binary, staffs, notes):
        self.binary = binary
        self.staffs = staffs
        self.notes = notes

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
        for note in self.notes:
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


def in_tolerance(value, target, tolerance):
    percent = abs(value - target) / target

    return percent < tolerance
