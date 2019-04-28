'''
'''
from pathlib import Path
import unittest

from vvc import config
from vvc.detector import faster_rcnn, yolo_v3
from vvc.vvc import VVC

class OtherTestCase(unittest.TestCase):
    
    tm_videos = ['Ch4_20181121071359_640x480.mp4', 'Ch4_20181121073138_640x480.mp4']


    def setUp(self):
        config.video_folder = str(Path(config.base_folder).joinpath('Videos').joinpath('Otros'))


    def tearDown(self):
        pass

    
    def test_yolo_naive_tm_person(self):
        
        detector = yolo_v3.YOLOV3('YOLOv3')
        
        counter = VVC(detector)
        
        for video_name in self.tm_videos:
            print(config.video_folder)
            print(video_name)
            counter.count(video_name, 
                          frame_rate_factor=1, 
                          filter_tags=['person'],
                          show_obj_id=False)
    
    