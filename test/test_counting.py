'''
'''
import unittest

from vvc import config, vvc
from vvc.detector import faster_rcnn, yolo_v3

class CountingTestCase(unittest.TestCase):
    
    video_name = 'MOV_0861.mp4'


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_faster_rcnn_naive(self):
        for model_name in config.models:
            if 'frcnn' in model_name:
                detector = faster_rcnn.FasterRCNN(model_name)
            
                vvc.VVC(self.video_name, detector).count()
        pass
    
    def test_yolo_naive(self):
        for model_name in config.models:
            if 'yolo' in model_name:
                detector = yolo_v3.YOLOV3(model_name)
            
                vvc.VVC(self.video_name, detector).count()
        pass
    
    