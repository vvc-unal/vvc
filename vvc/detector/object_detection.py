'''
'''
from enum import Enum

from vvc.detector import faster_rcnn, yolo_v3, retinanet

class Detector(Enum):
    FRCNN = 'frcnn-resnet50'
    FRCNN_TRANSFER = 'frcnn-resnet50-transfer'
    RETINANET = 'RetinaNet-ResNet50'
    YOLO3 = ['YOLOv3', 'yolo3']
    YOLO3_TRANSFER = ['YOLOv3-transfer', 'yolo3']
    YOLO3_PRUNNED = ['yolo3-prunned', 'yolo3']
    TINY_YOLO3 = ['YOLOv3-tiny', 'tiny']
    TINY_YOLO3_TRANSFER = ['YOLOv3-tiny-transfer', 'tiny']
    VVC1 = ['vvc1-yolov3', 'vvc1']
    VVC2 = ['vvc2-yolov3', 'vvc2']
    VVC3 = ['vvc3-yolov3', 'vvc3']


all_models = [Detector.FRCNN, Detector.FRCNN_TRANSFER,
          Detector.YOLO3, Detector.YOLO3_TRANSFER, 
          Detector.TINY_YOLO3, Detector.TINY_YOLO3_TRANSFER,
          Detector.VVC1, Detector.VVC2, Detector.VVC3,
          Detector.RETINANET]

basic_models = [Detector.FRCNN, Detector.FRCNN_TRANSFER, 
                Detector.YOLO3, Detector.YOLO3_TRANSFER, Detector.YOLO3_PRUNNED,
                Detector.TINY_YOLO3, Detector.TINY_YOLO3_TRANSFER,
                Detector.RETINANET]

frcnn_models = [Detector.FRCNN, Detector.FRCNN_TRANSFER]

yolo3_models = [Detector.YOLO3, Detector.YOLO3_TRANSFER, Detector.YOLO3_PRUNNED]

tiny_yolo3_models = [Detector.TINY_YOLO3, Detector.TINY_YOLO3_TRANSFER]

vvc_models = [Detector.VVC1, Detector.VVC2, Detector.VVC3]

__yolo_based_models = vvc_models + yolo3_models + tiny_yolo3_models

def get_detector(model):
    '''
    Returns and instance of a object detector.
    '''
    if model in frcnn_models:
        return faster_rcnn.FasterRCNN(model.value)
    elif model in __yolo_based_models:
        model_name = model.value[0]
        body_name = model.value[1]
        return yolo_v3.YOLOV3(model_name, body_name)
    elif model == Detector.RETINANET:
        return retinanet.RetinaNet(model.value)
    
    