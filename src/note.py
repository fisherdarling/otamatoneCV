from loc import Loc
import cv2


class Note(Loc):
    def __init__(self, acc=None, head=None, stem=None, dot=None, centroid=None, bounding_box=None):
        super().__init__(centroid, bounding_box)
        self.acc = acc
        self.head = head
        self.stem = stem
        self.dot = dot

    def draw(self, img):
        bb = self.bounding_box()
        cv2.rectangle(img, (bb[0], bb[1]), (bb[0] + bb[2],
                                            bb[1] + bb[3]), (255, 255, 0), thickness=2)

        if self.head:
            self.head.draw(img)

        if self.stem:
            self.stem.draw(img)

    def __repr__(self):
        return f"Note({self.acc},{self.head},{self.stem},{self.dot})"
