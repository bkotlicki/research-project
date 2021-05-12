import cv2


class IntersectionOverUnion:
    def __init__(self, boxA, boxB):
        self.boxA = boxA
        self.boxB = boxB

        self.intersection = self.calculate_intersection()
        self.union = self.calculate_union()

        self.iou = self.calculate_result()

    def calculate_intersection(self):
        top_left_x = max(self.boxA[0], self.boxB[0])
        top_left_y = max(self.boxA[1], self.boxB[1])

        bottom_right_x = min(self.boxA[0] + self.boxA[2], self.boxB[0] + self.boxB[2])
        bottom_right_y = min(self.boxA[1] + self.boxA[3], self.boxB[1] + self.boxB[3])

        w = max(0, bottom_right_x - top_left_x)
        h = max(0, bottom_right_y - top_left_y)

        return max(0, w * h)

    def calculate_union(self):
        box_a_area = self.boxA[2] * self.boxA[3]
        box_b_area = self.boxB[2] * self.boxB[3]

        return max(0, box_a_area + box_b_area - self.intersection)

    def calculate_result(self):
        if self.union == 0:
            return 0.0
        else:
            return self.intersection / self.union

    def result(self):
        return self.iou
