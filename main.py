from vvc import config, vvc
from vvc.detector import faster_rcnn

if __name__ == '__main__':
    
    for model_name in config.models:
        detector = faster_rcnn(model_name)
        vvc.VVC('MOV_0861.mp4', detector).count()
    