from vvc import config, vvc
from vvc.detector import faster_rcnn, yolo_v3

if __name__ == '__main__':
    
    for model_name in config.models:
        if 'frcnn' in model_name:
            #detector = faster_rcnn.FasterRCNN(model_name)
            continue
        elif 'yolo' in model_name:
            detector = yolo_v3.YOLOV3(model_name)
            
        vvc.VVC('CL 53 X CRA 60 910-911.mp4', detector).count()
        
    #vvc.VVC('MOV_0861.mp4', yolo_v3.YOLOV3('yolov3-transfer')).count()