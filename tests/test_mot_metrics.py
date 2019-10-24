'''
'''
import logging
from pathlib import Path 
import unittest

from vvc.format import cvat
from vvc import config

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class MOTMetricsTestCase(unittest.TestCase):
    
    train_video = 'MOV_0861'
    test_video = 'CL 53 X CRA 60 910-911'
    

    def setUp(self):
        pass


    def tearDown(self):
        pass

    def test_cvat_to_motchallenge_ground_truth(self):
        
        cvat_ground_truth_folder = Path(config.video_folder).joinpath('cvat')
            
        mot_challenge_folder = Path(config.video_folder)\
                                        .joinpath('motchallenge')
        
        mot_challenge_folder.mkdir(exist_ok = True)
        
        assert cvat_ground_truth_folder.exists()
        assert mot_challenge_folder.exists()
        
        for video_name in [self.train_video, self.test_video]:
            
            logging.info("Convert to MOT Challenge {}".format(video_name))
        
            cvat_file = cvat_ground_truth_folder.joinpath(video_name + '.xml')
            mot_challenge_file = mot_challenge_folder.joinpath(video_name + '.txt')
            
            assert cvat_file.exists() 
            
            # Transform
            
            df = cvat.to_mot_challenge(cvat_file, mot_challenge_file)
            
            logging.debug(df)
            
            assert mot_challenge_file.exists() 
            
    
    
                
    