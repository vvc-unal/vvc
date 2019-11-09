'''
'''
import logging
import unittest

from vvc.detector import object_detection, yolo_v3, retinanet
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
        
        for video_name in self.train_videos:
            vvc.count(video_name, frame_rate_factor=0.5)
            break
            
        '''for video_name in self.test_videos:
            vvc.count(video_name)
            break'''

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
        detector = retinanet.RetinaNet('RetinaNet-ResNet50')
        self.counting(detector)
                
    