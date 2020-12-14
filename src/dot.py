from loc import Loc
import cv2


class Dot(Loc):
    def __init__(self, centroid=None, bounding_box=None):
        super().__init__(centroid, bounding_box)

    def draw(self, img):
        bb = self.bounding_box()
        cv2.rectangle(img, (bb[0], bb[1]), (bb[0] + bb[2],
                                            bb[1] + bb[3]), (0, 0, 255), thickness=2)
        cv2.putText(img, "Dot", (bb[0], bb[1] - bb[3] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=1)

    def __repr__(self):
        return "DOT"
