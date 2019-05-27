'''
'''
from pathlib import Path
import unittest

from vvc import config
from vvc.detector import faster_rcnn, yolo_v3
from vvc.vvc import VVC
from tests.test_video import output_folder

class OtherTestCase(unittest.TestCase):
    
    tm_videos = ['Ch4_20181121071359_640x480.mp4', 'Ch4_20181121073138_640x480.mp4']
    tm_folder = str(Path(config.base_folder).joinpath('Videos').joinpath('Otros'))

    def setUp(self):
        pass

    def tearDown(self):
        pass

    
    def test_yolo_naive_tm_person(self):
        config.video_folder = self.tm_folder
        
        detector = yolo_v3.YOLOV3('YOLOv3')
        
        counter = VVC(detector)
        
        for video_name in self.tm_videos:
            print(config.video_folder)
            print(video_name)
            counter.count(video_name, 
                          frame_rate_factor=1, 
                          filter_tags=['person'],
                          show_obj_id=False)
            
    def test_yolo_naive_tm_workers(self):
        
        tm_workers_folder = Path(config.base_folder).joinpath('Videos/TM/TrabajadoresYPolicias')
        video_filder = str(tm_workers_folder.joinpath('Videos'))
        output_folder = str(tm_workers_folder.joinpath('vvc'))
        tm_workers_videos = ['Ch1_20181113075540_1min.mp4', 'Ch2_20181110121206_1min.mp4',
                             'Ch2_20181112171900_1min.mp4', 'Ch2_20181113171816_1min.mp4',
                             'Ch3_20181115065141_1min.mp4', 'Ch4_20181117115137_1min.mp4',
                             'Ch4_20181119065543_1min.mp4', 'Ch4_20181119164606_1min.mp4']
        
        config.video_folder = video_filder
        config.output_folder = output_folder
        detector = yolo_v3.YOLOV3('TM-YOLOv3')
        
        counter = VVC(detector)
        
        for video_name in tm_workers_videos:
            print(config.video_folder)
            print(video_name)
            counter.count(video_name, 
                          frame_rate_factor=1, 
                          filter_tags=['tu_llave', 'seg'],
                          show_obj_id=False)
    
    
