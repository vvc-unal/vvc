'''
'''
import logging
import os

import motmetrics as mm
from pathlib import Path
import unittest

from vvc.format import cvat, vvc_format
from vvc import config
from vvc.detector.object_detection import Model

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


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
                                        .joinpath('mot_challenge')
        
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
            
    def test_vvc_to_motchallenge(self):
        
        for video_name in [self.train_video, self.test_video]:
            
            # Set video folder
            vvc_folder = Path(config.output_folder).joinpath(video_name)
            
            assert vvc_folder.exists()
            
            mot_challenge_folder = vvc_folder.joinpath('mot_challenge')
            
            mot_challenge_folder.mkdir(exist_ok = True)
            
            # Select files
            
            for model in [Model.TINY_YOLO3, Model.RETINANET]:
                file_name = model.value
                
                vvc_file = vvc_folder.joinpath(file_name + '.mp4.json')
                mot_challenge_file = mot_challenge_folder.joinpath(file_name + '.txt')
                
                assert vvc_file.exists() 
            
            # Transform
            
            logging.info("Convert from VVC to MOT Challenge {}, variant: {}".format(video_name, file_name))
            
            df = vvc_format.to_mot_challenge(vvc_file, mot_challenge_file)
            
            logging.debug(df)
            
            assert mot_challenge_file.exists()
    
    def test_motchallenge_files(self):
        
        mot_challenge_folder = Path(config.video_folder).joinpath('mot_challenge')
        
        vvc_folder = Path(config.output_folder).joinpath(self.test_video).joinpath('mot_challenge')
        
        dnames = [
            'RetinaNet-ResNet50',
            'YOLOv3-tiny',
        ]
        
        def compute_motchallenge(dname):
            df_gt = mm.io.loadtxt(os.path.join(mot_challenge_folder, self.test_video + '.txt'))
            df_test = mm.io.loadtxt(os.path.join(vvc_folder, dname + '.txt'))
            return mm.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)
    
        accs = [compute_motchallenge(d) for d in dnames]
    
        # For testing
        #[a.events.to_pickle(n) for (a,n) in zip(accs, dnames)]
    
        mh = mm.metrics.create()
        summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=dnames, generate_overall=False)
    
        print()
        print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names, formatters=mh.formatters))
    
    
    