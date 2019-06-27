
from vvc.detector import faster_rcnn,yolo_v3
from vvc.vvc import VVC

if __name__ == '__main__':
    
    detector = yolo_v3.YOLOV3('YOLOv3-tiny-transfer')
    
    #detector = faster_rcnn.FasterRCNN('frcnn-resnet50-tunned')
    
    test1_video = 'MOV_0861'
    
    test2_video = 'CL 53 X CRA 60 910-911'
            
    VVC(detector).count( test1_video + '.mp4', frame_rate_factor=1, show_obj_id=True)