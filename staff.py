import cv2
import random
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def cvt_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def otsu_filter(img):
    _, img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


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

    # if the image is so small that it doesn't allow us to grab a random NUM_SELECTED_COL number of columns from the center 50%, then we will just not process this image
    if len(possible_cols) < NUM_SELECTED_COL:
        exit()

    random_col = random.sample(possible_cols, NUM_SELECTED_COL)

    black_modes = []
    white_modes = []

    for col in random_col:
        black_runs, white_runs = extract_runs(frame, col)
        black_modes += list(stats.mode(black_runs)[0])
        white_modes += list(stats.mode(white_runs)[0])

    staffline_height = int(np.median(black_modes))
    staffspace_height = int(np.median(white_modes))

    return (staffline_height, staffspace_height)


def potential_staff_line_positions(inverted):
    row_sums = np.sum(inverted, axis=1) / 255
    highest = np.max(row_sums)
    avg = int(np.average(np.where(row_sums > 0)))

    staff_lines = []
    current_line = []
    for row, line in enumerate(row_sums):
        # We are at a staff line
        if line > avg:
            current_line.append(row)
        # We are not at a staff line,
        # append if the current line
        # is not empty
        elif len(current_line) > 0:
            staff_lines.append(current_line)
            current_line = []

    return staff_lines


def remove_staff_lines(inverted, staff_lines, staffline_height):
    removed = inverted.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, int(staffline_height * 1.5)))

    for line in staff_lines:
        start_row = line[0] - staffline_height // 2
        end_row = line[-1] + staffline_height // 2

        eroded = cv2.erode(removed[start_row:end_row, :], kernel, iterations=1)
        removed[start_row:end_row, :] = eroded

    # eroded = cv2.erode(removed, kernel, iterations=1)
    return removed


def remove_vertical_lines(img, staffline_height):
    removed = img.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (staffline_height * 3, 1))

    img = cv2.erode(img, kernel)

    # kernel = cv2.getStructuringElement(
    #     cv2.MORPH_RECT, (31 // 2, 4)
    # )

    # img = cv2.dilate(img, kernel)

    return img


def detect_circles(binary, img, staffspace_height):
    # print(type(binary))
    # print(type(img))

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    note_heads = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        # print(perimeter)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi*(area / (perimeter * perimeter))

        # print(circularity)
        if circularity > 0.5:
            note_heads.append(contour)
        else:
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            new_perm = cv2.arcLength(approx, True)
            new_area = cv2.contourArea(approx)

            new_circ = 4 * np.pi*(new_area / (new_perm * new_perm))

            print(circularity, new_circ)

            if new_circ > 0.5:
                note_heads.append(approx)

    # print(note_heads)

    return cv2.drawContours(img, note_heads, -1, (0, 0, 255), 3)

    # return cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

    # circles = cv2.HoughCircles(
    #     gray, cv2.HOUGH_GRADIENT, 1, staffspace_height // 2, param1=50, param2=30, minRadius=staffspace_height // 3, maxRadius=staffspace_height * 2)

    # circles = np.round(circles[0, :]).astype("int")

    # for (x, y, r) in circles:
    #     cv2.circle(img, (x, y), r, (0, 255, 0), 4)
    #     cv2.rectangle(img, (x - 5, y - 5),
    #                   (x + 5, y + 5), (0, 128, 255), -1)


def main():
    # data = [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1]

    # (black_runs, white_runs) = extract_runs(data, 0)

    # print(f"{black_runs=}")
    # print(f"{white_runs=}")

    img = cv2.imread("ode-to-joy-cropped.png")
    cv2.imwrite("original.png", img)
    gray = cvt_gray(img)
    binary_music = otsu_filter(gray)
    height, width = binary_music.shape

    (staffline_height, staffspace_height) = extract_staff_metrics(binary_music)

    print(f"{binary_music.shape=}")
    print(f"{staffline_height=}")
    print(f"{staffspace_height=}")

    # inverted_music =
    inverted_music = cv2.bitwise_not(binary_music * 255)
    staff_lines = potential_staff_line_positions(inverted_music)
    no_staff_lines = remove_staff_lines(
        inverted_music, staff_lines, staffline_height)

    linesP = cv2.HoughLinesP(no_staff_lines, 0.5, np.pi /
                             180, 20, minLineLength=3 * staffspace_height, maxLineGap=staffspace_height)
    lines_img = img.copy()

    if linesP is not None:
        for i in range(0, len(linesP)):
            x1, y1, x2, y2 = linesP[i][0]

            if abs(x1 - x2) < 5:
                cv2.line(lines_img, (x1, y1),
                         (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

    removed_vertical = remove_vertical_lines(no_staff_lines, staffline_height)
    contoured = detect_circles(removed_vertical, img,  staffspace_height)

    cv2.namedWindow("Music", cv2.WINDOW_NORMAL)
    # cv2.imshow("Music", contoured)
    # cv2.imshow("Music", removed_vertical)
    # cv2.imshow("Music", cv2.bitwise_not(no_staff_lines))
    cv2.imshow("Music", lines_img)

    cv2.imwrite("removed_vertical_black.png", removed_vertical)
    cv2.imwrite("staff_line_removal.png", cv2.bitwise_not(no_staff_lines))
    cv2.imwrite("removed_vertical.png", cv2.bitwise_not(removed_vertical))
    cv2.imwrite("contoured.png", contoured)
    cv2.imwrite("contoured_with_lines.png",
                cv2.bitwise_or(contoured, lines_img))
    cv2.waitKey()


if __name__ == "__main__":
    main()
