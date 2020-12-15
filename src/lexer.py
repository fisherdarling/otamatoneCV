import matplotlib.pyplot as plt
import numpy as np
import cv2

from head import HeadType
from stem import StemType, Stem


TONES = ['B', 'C', 'D', 'E', 'F', 'G', 'A']
NUM_TONES = 7


class Lexer:
    def __init__(self, binary, staffs, notes, staffspace_height, staffline_height):
        self.binary = binary
        self.staffs = staffs
        self.notes = notes
        self.staffspace_height = staffspace_height
        self.staffline_height = staffline_height

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

            # If the stem is None and the headtype is filled, something is up
            if note.stem is None and note.head.type == HeadType.FILLED:
                n_x, n_y, n_w, n_h = note.bounding_box()

                # print(
                #     f"WEIRD: {note.head.centroid()}, {note.bounding_box()}, {self.staffspace_height}")

                # This is probably a note, must have missed
                # a stem, we'll artificially insert one:
                if n_h > self.staffspace_height * 3 and n_h < self.staffspace_height * 5:
                    print("Creating stem")

                    a = (n_x + n_w - self.staffline_height, n_y + 3)
                    b = (n_x + n_w - self.staffline_height,
                         n_y + n_h - note.head.bounding_box()[3] // 2)

                    stem = Stem(a=a, b=b)
                    note.stem = stem
                # Doesn't have a missed stem, delete it:
                else:
                    print("Deleting")
                    to_delete.append(i)
                    continue

            staff = self.nearest_staff(note)

            # Calculate the note head center:
            c_y = y + (h / 2)

            check_y = (c_y - staff.center()) / \
                ((self.staffspace_height + self.staffline_height) / 2)
            idx = -1 * round(check_y)

            print(f"[{staff.center()}]", c_y, check_y, "->", idx)

            if abs(idx) > 16:
                to_delete.append(i)
                continue

            note.tone = TONES[idx % NUM_TONES]

        while len(to_delete) > 0:
            self.notes.pop(to_delete.pop())

    def classify_stems(self):
        delta = int(self.staffline_height * 1.4)

        for note in self.notes:
            if note.stem is None:
                continue

            # note_aabb = note.bounding_box()
            n_x, n_y, n_w, n_h = note.bounding_box()
            x_min, x_max, y_min, y_max = note.stem.bb()

            # Probably has no stem continuation:
            # if x_max < (note_aabb[0] + note_aabb[2]) * 1.1:
            #     note.stem.type = StemType.NORMAL
            #     continue

            note_slice = self.binary[n_y:n_y + n_h, n_x:n_x + n_w]

            x_min -= delta
            x_max += delta
            y_min -= delta
            y_max += delta

            stem_w = x_max - x_min
            x_start = x_min - n_x

            print(f"{note_slice.shape}", (n_x, n_y), x_min, x_max,
                  "|", 0, n_h, x_start, x_start + stem_w, "|", note)

            slice = note_slice[0:n_h, x_start:x_start + stem_w]

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (x_max - x_min, 2))

            note_slice[0:n_h, x_start:x_start + stem_w] = cv2.erode(slice,
                                                                    kernel, iterations=2)

            # cv2.imshow("Music", note_slice)
            # cv2.waitKey()

            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(
                note_slice, 8, cv2.CV_32S)

            # Only the note head
            if cnt == 1:
                note.stem.type = StemType.NORMAL
                continue

            # num_components = filter(lambda idx: )
            count = 0
            for idx in range(cnt):
                x, y, w, h, a = get_bb(idx, stats)
                # c = centroids[idx]
                c = [x + w // 2, y + h // 2]

                # If the centroid is on the right side (skipping note_head)
                # if c[0] > x_start + stem_w:
                #     count += 1
                # print(c, w, h)
                # If the centroid is above the note_head
                if c[1] < (y_max - y_min) // 2 and w > n_w / 2.3 \
                        or c[0] > x_start + stem_w:
                    # cv2.imshow("Centroid", note_slice)
                    # cv2.waitKey(500)
                    count += 1
                    # print(c, count, y_max - y_min)

            # print(f"({n_x}, {n_y}) [{cnt}] {count}")

            if count == 0:
                note.stem.type = StemType.NORMAL
            elif count == 1:
                if note.is_beam:
                    note.stem.type = StemType.EIGHTH_BEAM
                else:
                    note.stem.type = StemType.EIGHTH_FLAG
            elif count == 2:
                if note.is_beam:
                    note.stem.type = StemType.SIXTEENTH_BEAM
                else:
                    note.stem.type = StemType.SIXTEENTH_FLAG
            else:
                # print("Unknown stem count:", count)
                note.stem.type = StemType.NORMAL

        # self.no_stems = removed

    def nearest_staff(self, note):
        _, c_y = note.centroid()

        return min(self.staffs, key=lambda staff: abs(staff.center() - c_y))

    def avg_staff_diff(self):
        total = 0.0

        for i in range(len(self.staffs) - 1):
            # print(self.staffs[i])

            total += (self.staffs[i + 1].center() - self.staffs[i].center())

        return total / len(self.staffs)

    def populate_notes(self):
        staff = self.staffs[self.current_staff]
        avg_dist = self.avg_staff_diff()

        self.next_notes = list(filter(
            lambda note: note.within_staff(staff, avg_dist), self.notes))
        self.next_notes.sort(key=lambda note: note.bounding_box()[0])

        self.note_idx = 0

    def __iter__(self):
        self.current_staff = 0
        self.populate_notes()

        return self

    def __next__(self):
        if self.note_idx >= len(self.next_notes):
            self.current_staff += 1

            if self.current_staff >= len(self.staffs):
                raise StopIteration

            self.populate_notes()

        self.note_idx += 1
        return self.next_notes[self.note_idx - 1]

    def fits_note_size(self, width, height):
        sh = self.staffspace_height + self.staffline_height * 2

        return width < sh * 3 and width > sh / 1.2 \
            and height < sh * 5 and height > sh / 1.5


def get_bb(idx, stats):
    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    w = stats[idx, cv2.CC_STAT_WIDTH]
    h = stats[idx, cv2.CC_STAT_HEIGHT]
    a = stats[idx, cv2.CC_STAT_AREA]

    return x, y, w, h, a


def in_tolerance(value, target, tolerance):
    percent = abs(value - target) / target

    return percent < tolerance
