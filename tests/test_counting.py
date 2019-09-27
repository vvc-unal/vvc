'''
'''
import unittest

from vvc import config
from vvc.detector import faster_rcnn, yolo_v3, retinanet
from vvc.vvc import VVC

class VVCTestCase(unittest.TestCase):
    
    train_videos = ['MOV_0861.mp4', 'MOV_0841.mp4', 'MOV_0866.mp4', 'MOV_0872.mp4']
    test_videos = ['CL 26 X CRA 33 600-610.mp4', 'CL 53 X CRA 60 910-911.mp4', 'CRA 7 X CL 45 955-959.mp4']


    def setUp(self):
        pass


    def tearDown(self):
        pass

    def counting(self, detector):
        vvc = VVC(detector)
        
        for video_name in self.train_videos:
            vvc.count(video_name, frame_rate_factor=0.1)
            
        for video_name in self.test_videos:
            vvc.count(video_name)

    def test_faster_rcnn_naive(self):
        for model_name in config.models:
            if 'frcnn' in model_name:
                detector = faster_rcnn.FasterRCNN(model_name)
                self.counting(detector)
    
    def test_yolo_naive(self):
        detector = yolo_v3.YOLOV3('YOLOv3')
        self.counting(detector)
        
    def test_yolo_transfer_naive(self):
        detector = yolo_v3.YOLOV3('YOLOv3-transfer')
        self.counting(detector)
        
    def test_yolo_tiny_pretrained_naive(self):
        detector = yolo_v3.YOLOV3('YOLOv3-tiny')
        self.counting(detector)
        
    def test_retinanet_naive(self):
        detector = retinanet.RetinaNet('RetinaNet-ResNet50')
        self.counting(detector)
                
    