from loc import Loc
import cv2


class Note(Loc):
    def __init__(self, acc=None, head=None, stem=None, dot=None, tone=None, centroid=None, bounding_box=None):
        super().__init__(centroid, bounding_box)
        self.acc = acc
        self.head = head
        self.stem = stem
        self.dot = dot
        self.tone = tone
        self.is_beam = False

        if not tone:
            self.tone = "UNK"

    def draw(self, img):
        bb = self.bounding_box()
        cv2.rectangle(img, (bb[0], bb[1]), (bb[0] + bb[2],
                                            bb[1] + bb[3]), (255, 255, 0), thickness=2)

        if self.head:
            self.head.draw(img)

        if self.stem:
            self.stem.draw(img)

        if self.tone:
            cv2.putText(img, self.tone, (bb[0], bb[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), thickness=2)

        if self.dot:
            self.dot.draw(img)

    def within_staff(self, staff, avg_dist):
        _, y, _, h = self.bounding_box()
        c_y = y + h / 2

        return abs(staff.center() - c_y) < avg_dist / 2

    def __repr__(self):
        if self.head:
            return f"Note({self.tone}, {self.head}, {self.stem}, {self.dot if self.dot else 'NO_DOT'})"
