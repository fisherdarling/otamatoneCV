

class Loc:
    def __init__(self, centroid=None, bounding_box=None):
        self.c = centroid
        self.aabb = bounding_box

    def centroid(self):
        return self.c

    def bounding_box(self):
        return self.aabb

    def bounding_box_area(self):
        _, _, w, h = self.bounding_box()

        return w * h


# Taken from: https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
# a and b are in the form: x, y, w, h
def intersection_area(a, b):
    dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if (dx >= 0) and (dy >= 0):
        return dx*dy
