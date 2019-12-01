
class Tracker:

    def __init__(self):
        self.last_tags_id = {}

    @staticmethod
    def iou(a_box, b_box):
        '''
        Check if the boxes are overlapping
        '''
        (ax0, ay0, ax1, ay1) = a_box
        (bx0, by0, bx1, by1) = b_box

        x0 = max(ax0, bx0)
        y0 = max(ay0, by0)
        x1 = min(ax1, bx1)
        y1 = min(ay1, by1)

        inter_area = max(0, x1 - x0) * max(0, y1 - y0)

        a_area = (ax1 - ax0) * (ay1 - ay0)
        b_area = (ax1 - ax0) * (ay1 - ay0)

        iou = inter_area / (a_area + b_area - inter_area)

        return iou

    def get_next_id(self, tag):
        '''
        Get the next id for the Tag
        '''
        if tag in self.last_tags_id:
            self.last_tags_id[tag] += 1
        else:
            self.last_tags_id[tag] = 1
        return tag + " " + str(self.last_tags_id[tag])
