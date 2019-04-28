
from vvc.detector import yolo_v3
from vvc.vvc import VVC

if __name__ == '__main__':
    
    detector = yolo_v3.YOLOV3('yolov3')
            
    VVC(detector).count('MOV_0861.mp4', filter_tags=['person'], show_obj_id=False)