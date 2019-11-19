'''
'''
import logging
import unittest

from vvc.detector import object_detection, retinanet
from vvc.tracker.iou_tracker import IOUTracker
from vvc.vvc import VVC

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class VVCTestCase(unittest.TestCase):
    
    videos = ['MOV_0861.mp4', 'CL 53 X CRA 60 910-911.mp4']

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def counting(self, detector):
        tracker = IOUTracker(iou_threshold=0.5, dectection_threshold=0.8, min_track_len=4, patience=2)
        vvc = VVC(detector, tracker)
        
        for video_name in self.videos:
            vvc.count(video_name, frame_rate_factor=0.2)

    def test_faster_rcnn_naive(self):
        logging.info('faster_rcnn')
        for model in object_detection.frcnn_models:
            logging.info(model.value)
            detector = object_detection.get_detector(model)
            self.counting(detector)
    
    def test_yolo_naive(self):
        logging.info('yolo v3')
        for model in object_detection.yolo3_models:
            print('Detector: {}'.format(model))
            detector = object_detection.get_detector(model)
            self.counting(detector)
        
    def test_yolo_tiny_naive(self):
        logging.info('yolo v3 tiny')
        
        for model in object_detection.tiny_yolo3_models:
            detector = object_detection.get_detector(model)
            self.counting(detector)

    def test_vvc_naive(self):
        logging.info('vvc')
        for model in object_detection.vvc_models:
            detector = object_detection.get_detector(model)
            
            self.counting(detector)
        
    def test_retinanet_naive(self):
        logging.info('retinanet')
        detector = object_detection.get_detector(object_detection.Detector.RETINANET)
        self.counting(detector)

