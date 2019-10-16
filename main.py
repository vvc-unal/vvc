import logging

from vvc.detector.object_detection import get_detector, Model
from vvc.vvc import VVC

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

if __name__ == '__main__':
    
    detector = get_detector(Model.RETINANET)
    
    test_1_video = 'MOV_0861'
    
    test_2_video = 'CL 53 X CRA 60 910-911'
            
    VVC(detector).count( test_1_video + '.mp4', frame_rate_factor=0.5)