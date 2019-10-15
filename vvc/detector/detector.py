'''
'''
__frcnn = 'frcnn-resnet50'
__frcnn_transfer = 'frcnn-resnet50-transfer'
__retinanet = 'RetinaNet-ResNet50'
__yolo3 = 'YOLOv3'
__yolo3_transfer = 'YOLOv3-transfer'
__tiny_yolo3 = 'YOLOv3-tiny'
__tiny_yolo3_transfer = 'YOLOv3-tiny-transfer'

__vvc1 = 'vvc1-yolov3'
__vvc2 = 'vvc2-yolov3'
__vvc3 = 'vvc3-yolov3'

all_models = [__frcnn, __frcnn_transfer,
          __yolo3, __yolo3_transfer, 
          __tiny_yolo3, __tiny_yolo3_transfer,
          __vvc1, __vvc2, __vvc3,
          __retinanet]

basic_models = [__frcnn, __frcnn_transfer, 
                __yolo3, __yolo3_transfer, 
                __tiny_yolo3, __tiny_yolo3_transfer,
                __retinanet]

def __init__(self, params):
    '''
    Constructor
    '''
    