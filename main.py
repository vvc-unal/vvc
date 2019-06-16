
from vvc.detector import faster_rcnn,yolo_v3
from vvc.vvc import VVC

if __name__ == '__main__':
    
    detector = yolo_v3.YOLOV3('YOLOv3')
    
    #detector = faster_rcnn.FasterRCNN('frcnn-resnet50-tunned')
            
    VVC(detector).count('CL 53 X CRA 60 910-911' + '.mp4', show_obj_id=True)