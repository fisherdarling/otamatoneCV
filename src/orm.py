import cv2
from staff import Staff, StaffLine
import numpy as np
from stem import Stem
import stem


def nothing(x):
    pass


class ORM:
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

        cv2.imshow("Music", self.inverted_music)
        cv2.waitKey()

        self.remove_staff_lines()

        cv2.imshow("Music", self.no_staff_lines)
        cv2.waitKey()

        self.find_hough_lines(
            20, 3 * self.staffspace_height, self.staffspace_height)
        self.find_probable_stems()

        cv2.createTrackbar('Votes', 'Music', 0,
                           50, nothing)
        cv2.createTrackbar('Min Length', 'Music', 0,
                           5 * self.staffspace_height, nothing)
        cv2.createTrackbar('Line Gap', 'Music', 0,
                           self.staffspace_height * 2, nothing)

        while(1):
            if cv2.waitKey(1) & 0xFF == 27:
                break

            votes = cv2.getTrackbarPos('Votes', 'Music')
            min_length = cv2.getTrackbarPos('Min Length', 'Music')
            gap = cv2.getTrackbarPos('Line Gap', 'Music')

            self.find_hough_lines(int(votes), int(min_length), int(gap))
            self.find_probable_stems()

            draw_img = cv2.cvtColor(self.inverted_music, cv2.COLOR_GRAY2BGR)
            self.draw_probable_stems(draw_img)

            cv2.imshow("Music", draw_img)
            # cv2.waitKey()

        # draw_img = self.inverted_music.copy() * 200.0
        # print(np.max(draw_img))

        return None

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
        linesP = cv2.HoughLinesP(self.no_staff_lines, 0.5, np.pi / 180, votes,
                                 minLineLength=min_length, maxLineGap=line_gap)

        self.hough_lines = linesP

    def find_probable_stems(self):
        self.probable_stems = []

        for line in self.hough_lines:
            line = line[0]
            angle = stem.line_angle(line[0], line[1], line[2], line[3])

            # print(angle)

            # Within 5 degrees of straight up and down:
            if abs((np.pi / 2) - angle) < 0.1:
                centroid = stem.line_center(line[0], line[1], line[2], line[3])
                new_stem = Stem(type=None, line=line, centroid=centroid)

                # print("Found Line")

                self.probable_stems.append(new_stem)

    def draw_probable_stems(self, img):
        for p_stem in self.probable_stems:
            line = p_stem.line
            # print(line)
            cv2.line(img, (line[0], line[1]),
                     (line[2], line[3]), (0, 255, 0), 1)


if __name__ == "__main__":
    orm = ORM("ode-to-joy-cropped.png")
    orm.extract_music()
