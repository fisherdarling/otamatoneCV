

class Loc:
    def __init__(self, centroid=None, bounding_box=None):
        self.centroid = centroid
        self.bounding_box = bounding_box

    def centroid(self):
        return self.centroid

    def bounding_box(self):
        return self.bounding_box
