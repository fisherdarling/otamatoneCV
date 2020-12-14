from loc import Loc
from enum import Enum
import cv2


class HeadType(Enum):
    FILLED = 1
    EMPTY = 2


class Head(Loc):
    def __init__(self, type=None, area=None, centroid=None, bounding_box=None):
        super().__init__(centroid, bounding_box)
        self.type = type
        self.area = area

    def from_cnt(cnt):
        x, y, w, h = cv2.boundingRect(cnt)

        M = cv2.moments(cnt)
        c_x = int(M["m10"] / M["m00"])
        c_y = int(M["m01"] / M["m00"])

        area = cv2.contourArea(cnt)

        return Head(type=None, area=area, centroid=(c_x, c_y), bounding_box=[x, y, w, h])

    def draw(self, img):
        bb = self.bounding_box()
        cv2.rectangle(img, (bb[0], bb[1]), (bb[0] + bb[2],
                                            bb[1] + bb[3]), (0, 0, 255), thickness=2)

        c_x, c_y = self.centroid()
        text = ""

        if self.type:
            text = f"{self.type.name}"
        else:
            text = f"{self.area}"

        cv2.putText(img, text, (c_x, c_y + bb[3] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=1)

    def covered_area(self):
        return self.area / self.bounding_box_area()

    def __eq__(self, other):
        # and self.bounding_box() == other.bounding_box()
        return self.centroid == other.centroid

    def __hash__(self):
        return hash(self.centroid)

    def __repr__(self):
        if self.type:
            return self.type.name
        else:
            return "UNK"
