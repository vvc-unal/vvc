'''
'''
from enum import Enum, unique

from vvc.detector import faster_rcnn, yolo_v3, retinanet


@unique
class Detector(Enum):
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


all_models = [Detector.FRCNN, Detector.FRCNN_TRANSFER,
          Detector.YOLO3, Detector.YOLO3_TRANSFER, Detector.YOLO3_PRUNNED, 
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
        detector_body = {Detector.YOLO3: 'yolo3', 
                         Detector.YOLO3_TRANSFER: 'yolo3', 
                         Detector.YOLO3_PRUNNED: 'yolo3',
                         Detector.TINY_YOLO3: 'tiny', 
                         Detector.TINY_YOLO3_TRANSFER: 'tiny',
                         Detector.VVC1: 'vvc1', 
                         Detector.VVC2: 'vvc2',
                         Detector.VVC3: 'vvc3'}
        
        model_name = model.value
        body_name = detector_body.get(model)
        
        return yolo_v3.YOLOV3(model_name, body_name)
    elif model == Detector.RETINANET:
        return retinanet.RetinaNet(model.value)
    
    