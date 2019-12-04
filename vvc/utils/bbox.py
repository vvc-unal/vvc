
def bbox_to_rectangle(bbox):
    return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]


def rectangle_to_bbox(rectangle):
    return [rectangle[0], rectangle[1], rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]]
