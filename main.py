from vvc import config, vvc
from vvc.detector import faster_rcnn, yolo_v3

if __name__ == '__main__':
    
    detector = yolo_v3.YOLOV3('yolov3')
            
    vvc.VVC('CL 53 X CRA 60 910-911.mp4', detector).count()