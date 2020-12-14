import cv2
from staff import Staff, StaffLine
import numpy as np
from stem import Stem
import stem
from note import Note
from head import Head, HeadType
from loc import *
from lexer import Lexer


def nothing(x):
    pass


class OMR:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        cv2.imwrite("original.png", self.img)
        self.height, self.width, _ = self.img.shape
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def extract_music(self):
        cv2.namedWindow("Music", cv2.WINDOW_NORMAL)

        self.otsu_filter()
        self.extract_staff_metrics()
        self.create_inverted()
        self.find_staff_lines()
        print(self.staffs)

        # cv2.imshow("Music", self.inverted_music)
        # cv2.waitKey()

        self.remove_staff_lines()

        # self.display_image(self.no_staff_lines)
        # cv2.imshow("Music", self.no_staff_lines)
        # cv2.waitKey()

        self.canny = cv2.Canny(self.no_staff_lines, 0, 1)

        # kernel = cv2.getStructuringElement(
        #     cv2.MORPH_RECT, (self.staffline_height * 2, self.staffline_height))
        # self.canny = cv2.dilate(self.canny, kernel)

        # self.canny = cv2.Laplacian(
        #     self.no_staff_lines, cv2.CV_64F, ksize=29)

        # cv2.imshow("Music", self.canny)
        # cv2.waitKey()

        # return

        # print(3 * self.staffspace_height, self.staffline_height)

        self.find_hough_lines(
            30, 3 * self.staffspace_height, self.staffspace_height * 5)
        self.find_probable_stems()

        # print(self.probable_stems)

        # draw_img = cv2.cvtColor(self.inverted_music, cv2.COLOR_GRAY2BGR)
        # self.draw_stems(draw_img, self.probable_stems)
        # self.display_image(draw_img)

        self.combine_probable_stems()

        draw_img = cv2.cvtColor(self.inverted_music, cv2.COLOR_GRAY2BGR)
        self.draw_stems(draw_img, self.probable_stems)
        # self.display_image(draw_img)

        cv2.imwrite("detected_stems.png", draw_img)

        self.detect_note_heads()
        self.determine_notes()

        draw_img = cv2.cvtColor(self.inverted_music, cv2.COLOR_GRAY2BGR)
        for note in self.probable_notes:
            note.draw(draw_img)
            # OMR.draw_bounding_box(draw_img, note, (255, 255, 0))
            # OMR.draw_bounding_box(draw_img, note.head, (0, 0, 255))
            # OMR.draw_bounding_box(draw_img, note.stem, (0, 255, 0))

        # print(self.probable_notes)

        # self.display_image(draw_img)
        print("Here")
        cv2.imwrite("detected_notes.png", draw_img)

        lexer = Lexer(self.inverted_music, self.staffs, self.probable_notes)
        lexer.generate_statistics()
        lexer.classify_heads()

        draw_img = cv2.cvtColor(self.inverted_music, cv2.COLOR_GRAY2BGR)
        for note in self.probable_notes:
            note.draw(draw_img)

        self.display_image(draw_img)
        # cv2.imshow("Music", draw_img)

        return None

    def display_image(self, img):
        cv2.imshow("Music", img)
        cv2.waitKey()

    # def draw_bounding_box(img, l: Loc, color):
    #     if l is None:
    #         return

    #     bb = l.bounding_box()
    #     cv2.rectangle(img, (bb[0], bb[1]), (bb[0] + bb[2],
    #                                         bb[1] + bb[3]), color, thickness=2)

    def otsu_filter(self):
        _, img = cv2.threshold(
            self.gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.binary = img

    def extract_staff_metrics(self):
        self.staffline_height, self.staffspace_height = Staff.extract_staff_metrics(
            self.binary)

    def create_inverted(self):
        self.inverted_music = cv2.bitwise_not(self.binary * 255)

    def find_staff_lines(self):
        row_sums = np.sum(self.inverted_music, axis=1) / 255
        highest = np.max(row_sums)
        avg = int(np.average(np.where(row_sums > 0)))

        staffs = []
        last_staff_line = None
        current_line = []
        current_staff = Staff()

        for row, line in enumerate(row_sums):
            # We are at a staff line
            if line > avg:
                current_line.append(row)

            # We are not at a staff line,
            # append if the current line
            # is not empty
            elif len(current_line) > 0:
                next_staffline = StaffLine.from_ys(current_line)
                next_center = next_staffline.center()

                if last_staff_line is None or abs(next_center - last_staff_line.center()) < 2 * self.staffspace_height:
                    current_staff.add_line(next_staffline)
                else:
                    staffs.append(current_staff)
                    current_staff = Staff()
                    current_staff.add_line(next_staffline)

                last_staff_line = next_staffline
                current_line = []

        if current_staff.len() > 0:
            staffs.append(current_staff)

        self.staffs = staffs

    def remove_staff_lines(self):
        removed = self.inverted_music.copy()
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, int(self.staffline_height * 1.5)))

        for staff in self.staffs:
            for line in staff.lines:
                start_row = round(line.center()) - self.staffline_height
                end_row = round(line.center()) + self.staffline_height

                eroded = cv2.erode(
                    removed[start_row:end_row, :], kernel, iterations=1)
                removed[start_row:end_row, :] = eroded

        self.no_staff_lines = removed

    def find_hough_lines(self, votes, min_length, line_gap):
        linesP = cv2.HoughLinesP(self.canny, rho=1, theta=np.pi / 180, threshold=votes,
                                 minLineLength=min_length, maxLineGap=line_gap)

        self.hough_lines = linesP

    def find_probable_stems(self):
        self.probable_stems = []

        for line in self.hough_lines:
            line = line[0]
            angle = stem.line_angle(line[0], line[1], line[2], line[3])

            # print(angle)

            # Within ~5 degrees of straight up and down:
            if abs((np.pi / 2) - angle) < 0.1:
                centroid = stem.line_center(line[0], line[1], line[2], line[3])
                new_stem = Stem(line=line, centroid=centroid)
                # print(new_stem)
                # print("Found Line")

                self.probable_stems.append(new_stem)

    def combine_probable_stems(self):
        delta = self.avg_staff_diff() // 2

        start_len = len(self.probable_stems)
        while True:
            self.probable_stems.sort()
            new_stems = []

            for staff in self.staffs:
                y_top = staff.center() + delta
                y_bot = staff.center() - delta

                lines = list(filter(lambda stem: stem.y_max() <
                                    y_top and stem.y_min() > y_bot, self.probable_stems))

                new_stems.extend(Stem.combine_similar_lines(
                    lines, self.staffspace_height))

            self.probable_stems = new_stems
            if len(self.probable_stems) == start_len:
                break
            else:
                start_len = len(self.probable_stems)

    def avg_staff_diff(self):
        total = 0.0

        for i in range(len(self.staffs) - 1):
            # print(self.staffs[i])

            total += (self.staffs[i + 1].center() - self.staffs[i].center())

        return total / (len(self.staffs) - 1)

    def detect_note_heads(self):
        # return width < sh * 2 and width > sh / 5 \
        #     and height < sh * 1.2 and height > sh / 2

        # sh = self.staffspace_height + self.staffline_height * 2

        # print("Looking for these parameters:")
        # print(f"{self.staffspace_height / 5} < width < {self.staffspace_height * 2}")
        # print(f"{self.staffspace_height / 2} < height < {self.staffspace_height * 1.2}")

        self.remove_stems()
        # self.display_image(self.no_stems)
        cv2.imwrite("no_stems.png", self.no_stems)

        img = self.no_stems.copy()
        contours, _ = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        draw_img = cv2.cvtColor(self.no_staff_lines, cv2.COLOR_GRAY2BGR)
        note_heads_cnts = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(draw_img, (x, y), (x + w, y + h),
                          (255, 255, 0), thickness=2)

            cv2.putText(draw_img, f"w: {w}, h: {h}", (x, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=1)

            if self.fits_note_head(w, h):
                cv2.fillPoly(draw_img, pts=[cnt], color=(0, 0, 255))
                note_heads_cnts.append(cnt)

        self.draw_stems(draw_img, self.probable_stems)

        # cv2.drawContours(draw_img, note_heads_cnts, -1, (0, 0, 255), 2)
        # self.display_image(draw_img)

        cv2.imwrite("detected_note_heads.png", draw_img)

        self.note_head_cnts = note_heads_cnts
        self.probable_heads = list(map(
            lambda cnt: Head.from_cnt(cnt), self.note_head_cnts))

        to_remove = []
        for i in range(len(self.probable_heads) - 1):
            for j in range(i + 1, len(self.probable_heads)):
                head_a = self.probable_heads[i]
                head_b = self.probable_heads[j]

                inter_area = intersection_area(
                    head_a.bounding_box(), head_b.bounding_box())

                if inter_area and (in_tolerance(inter_area, head_a.area, 0.2) or in_tolerance(inter_area, head_b.area, 0.2)):
                    if head_a.area > head_b.area:
                        to_remove.append(j)
                        head_a.type = HeadType.EMPTY
                    else:
                        to_remove.append(i)
                        head_b.type = HeadType.Empty

        to_remove.sort()
        idxs = reversed(to_remove)

        for idx in idxs:
            self.probable_heads.remove(idx)

    def determine_notes(self):
        assert len(self.note_head_cnts)

        notes = []
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.no_staff_lines, 8, cv2.CV_32S)

        remaining_stems = set(self.probable_stems)
        remaining_heads = set(self.probable_heads)

        for i in range(cnt):
            x, y, w, h, a = get_bb(i, stats)

            if not self.fits_note_size(w, h):
                continue

            chosen_head = None
            for head in remaining_heads:
                # Is None if no intersection
                inter_area = intersection_area(
                    head.bounding_box(), [x, y, w, h])

                # if inter_area:
                #     print(head, inter_area, head.area)

                # If at least 80% of the head intersects with the CCA:
                if inter_area:
                    chosen_head = head
                    break

            # abort, this CCA is definitely not a note:
            if chosen_head is None:
                continue

            remaining_heads.remove(chosen_head)

            chosen_stem = None
            for stem in remaining_stems:
                # If the x_mid of the stem is near the head's x
                if abs(stem.x_center() - head.centroid()[0]) < self.staffspace_height:
                    # If the end of the stem is at about the head's midpoint
                    if abs(stem.y_max() - head.centroid()[1]) < self.staffspace_height / 2 \
                            or abs(stem.y_max() - head.centroid()[1]) < self.staffspace_height / 2:
                        chosen_stem = stem
                        break

                # if stem.intersects_rect(x, y, w, h) \
                #         and in_tolerance(stem.a[0], x + w, 0.3) \
                #         and in_tolerance(stem.a[1], y, 0.3):
                #     chosen_stem = stem
                #     break

            if chosen_stem is not None:
                remaining_stems.remove(chosen_stem)

            new_note = Note(head=chosen_head, stem=chosen_stem)
            new_note.centroid = centroids[i]
            new_note.aabb = [x, y, w, h]
            new_note.area = a

            notes.append(new_note)

            # notes.append()
            # new_note = Note()

        print(f"Remaining Stems Len: {len(remaining_stems)}")
        self.probable_notes = notes

    def remove_stems(self):
        removed = self.no_staff_lines.copy()

        height, width = removed.shape
        # print(f"width: {width}, height: {height}")
        delta = int(self.staffline_height * 1.4)

        for stem in self.probable_stems:
            # print(stem, stem.bb)

            x_min, x_max, y_min, y_max = stem.bb()
            x_min -= delta
            x_max += delta
            y_min -= delta
            y_max += delta

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (x_max - x_min, 2))

            eroded = cv2.erode(removed[y_min:y_max, x_min:x_max],
                               kernel, iterations=2)
            removed[y_min:y_max, x_min:x_max] = eroded

        self.no_stems = removed

    def fits_note_head(self, width, height):
        sh = self.staffspace_height + self.staffline_height * 2

        return width < sh * 2 and width > sh / 5 \
            and height < sh * 1.2 and height > sh / 1.5

    def fits_note_size(self, width, height):
        sh = self.staffspace_height + self.staffline_height * 2

        return width < sh * 3 and width > sh / 1.2 \
            and height < sh * 5 and height > sh / 1.5

    def draw_stems(self, img, lines):
        for stem in lines:
            cv2.line(img, stem.a,
                     stem.b, (0, 255, 0), 3)


def in_tolerance(value, target, tolerance):
    percent = abs(value - target) / target

    return percent < tolerance


def get_bb(idx, stats):
    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    w = stats[idx, cv2.CC_STAT_WIDTH]
    h = stats[idx, cv2.CC_STAT_HEIGHT]
    a = stats[idx, cv2.CC_STAT_AREA]

    return x, y, w, h, a


if __name__ == "__main__":
    omr = OMR("ode-to-joy-cropped.png")
    omr.extract_music()