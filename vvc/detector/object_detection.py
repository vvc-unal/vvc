'''
'''
from enum import Enum

from vvc.detector import faster_rcnn, yolo_v3, retinanet

class Model(Enum):
    FRCNN = 'frcnn-resnet50'
    FRCNN_TRANSFER = 'frcnn-resnet50-transfer'
    RETINANET = 'RetinaNet-ResNet50'
    YOLO3 = 'YOLOv3'
    YOLO3_TRANSFER = 'YOLOv3-transfer'
    YOLO3_PRUNNED = 'yolo3-prunned'
    TINY_YOLO3 = 'YOLOv3-tiny'
    TINY_YOLO3_TRANSFER = 'YOLOv3-tiny-transfer'
    VVC1 = 'vvc1-yolov3'
    VVC2 = 'vvc2-yolov3'
    VVC3 = 'vvc3-yolov3'


all_models = [Model.FRCNN, Model.FRCNN_TRANSFER,
          Model.YOLO3, Model.YOLO3_TRANSFER, 
          Model.TINY_YOLO3, Model.TINY_YOLO3_TRANSFER,
          Model.VVC1, Model.VVC2, Model.VVC3,
          Model.RETINANET]

basic_models = [Model.FRCNN, Model.FRCNN_TRANSFER, 
                Model.YOLO3, Model.YOLO3_TRANSFER, Model.YOLO3_PRUNNED,
                Model.TINY_YOLO3, Model.TINY_YOLO3_TRANSFER,
                Model.RETINANET]

frcnn_models = [Model.FRCNN, Model.FRCNN_TRANSFER]

yolo3_models = [Model.YOLO3, Model.YOLO3_TRANSFER, Model.YOLO3_PRUNNED]

tiny_yolo3_models = [Model.TINY_YOLO3, Model.TINY_YOLO3_TRANSFER]

vvc_models = [Model.VVC1, Model.VVC2, Model.VVC3]

__yolo_based_models = vvc_models + yolo3_models + tiny_yolo3_models

def get_detector(model):
    '''
    Returns and instance of a object detector.
    '''
    if model in frcnn_models:
        return faster_rcnn.FasterRCNN(model.value)
    elif model in __yolo_based_models:
        return yolo_v3.YOLOV3(model.value)
    elif model == Model.RETINANET:
        return retinanet.RetinaNet(model.value)
    
    