'''
'''
import unittest

from vvc import config
from vvc.detector import faster_rcnn, yolo_v3
from vvc.vvc import VVC

class VVCTestCase(unittest.TestCase):
    
    video_name = 'MOV_0861.mp4'


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_faster_rcnn_naive(self):
        for model_name in config.models:
            if 'frcnn' in model_name:
                detector = faster_rcnn.FasterRCNN(model_name)
            
                VVC(detector).count(self.video_name)
        pass
    
    def test_yolo_naive(self):
        for model_name in config.models:
            if 'yolo' in model_name:
                detector = yolo_v3.YOLOV3(model_name)
            
                VVC(detector).count(self.video_name)
        pass
    
    