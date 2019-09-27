
from vvc.detector import faster_rcnn,yolo_v3,retinanet
from vvc.vvc import VVC

if __name__ == '__main__':
    
    #detector = yolo_v3.YOLOV3('vvc3-yolov3')
    
    #detector = faster_rcnn.FasterRCNN('frcnn-resnet50-tunned')
    
    detector = retinanet.RetinaNet('RetinaNet-ResNet50')
    
    test_1_video = 'MOV_0861'
    
    test_2_video = 'CL 53 X CRA 60 910-911'
            
    VVC(detector).count( test_2_video + '.mp4', frame_rate_factor=1, show_obj_id=True)