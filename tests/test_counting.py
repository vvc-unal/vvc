'''
'''
import logging
import unittest

from vvc import config
from vvc.detector import faster_rcnn, yolo_v3, retinanet
from vvc.vvc import VVC


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class VVCTestCase(unittest.TestCase):
    
    train_videos = ['MOV_0861.mp4', 'MOV_0841.mp4', 'MOV_0866.mp4', 'MOV_0872.mp4']
    test_videos = ['CL 53 X CRA 60 910-911.mp4', 'CL 26 X CRA 33 600-610.mp4', 'CRA 7 X CL 45 955-959.mp4']


    def setUp(self):
        pass


    def tearDown(self):
        pass

    def counting(self, detector):
        vvc = VVC(detector)
        
        '''for video_name in self.train_videos:
            vvc.count(video_name, frame_rate_factor=0.5)
            break'''
            
        for video_name in self.test_videos:
            vvc.count(video_name)
            break

    def test_faster_rcnn_naive(self):
        logging.info('faster_rcnn')
        for model_name in config.models:
            if 'frcnn' in model_name:
                detector = faster_rcnn.FasterRCNN(model_name)
                self.counting(detector)
    
    def test_yolo_naive(self):
        logging.info('yolo v3')
        detector = yolo_v3.YOLOV3('YOLOv3')
        self.counting(detector)
        
    def test_yolo_transfer_naive(self):
        logging.info('yolo v3 transfer')
        detector = yolo_v3.YOLOV3('YOLOv3-transfer')
        self.counting(detector)
        
    def test_yolo_tiny_pretrained_naive(self):
        logging.info('yolo v3 tiny')
        detector = yolo_v3.YOLOV3('YOLOv3-tiny')
        self.counting(detector)
        
    def test_yolo_tiny_transfer_naive(self):
        detector = yolo_v3.YOLOV3('YOLOv3-tiny-transfer')
        self.counting(detector)
    
        
    def test_vvc_naive(self):
        logging.info('vvc')
        for model in ['vvc1-yolov3', 'vvc2-yolov3', 'vvc3-yolov3']:
            detector = yolo_v3.YOLOV3(model)
            
            self.counting(detector)
        
    def test_retinanet_naive(self):
        logging.info('retinanet')
        detector = retinanet.RetinaNet('RetinaNet-ResNet50')
        self.counting(detector)
                
    