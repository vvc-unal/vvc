'''
'''
from pathlib import Path

import numpy as np

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

from vvc.config import model_folder


class RetinaNet(object):
    '''
    classdocs
    '''
    
    labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                       7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                       12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
                       19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                       25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
                       31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                       36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                       41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                       48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                       54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                       60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                       66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                       72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                       78: 'hair drier', 79: 'toothbrush'}
    labels_to_names.update({3: 'motorbike'})

    def __init__(self, model_name):
        '''
        Constructor
        '''
        self.model_name = model_name
        self.labels_to_names = RetinaNet.labels_to_names
        
        # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
        model_path = Path(model_folder).joinpath(model_name).joinpath('resnet50_coco_best_v2.1.0.h5')
        
        # load retinanet model
        self.model = models.load_model(model_path, backbone_name='resnet50')
        
    def predict(self, frame):
        
        # preprocess image for network
        image = preprocess_image(frame)
        image, scale = resize_image(image)
        
        # process image
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        
        # correct for image scale
        boxes /= scale
        
        final_bboxes = []
        
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break
            
            bbox = {}
            bbox['class'] = self.labels_to_names[label]
            bbox['box'] = [int(i) for i in box]
            bbox['prob'] = score
            final_bboxes.append(bbox)
 
        return final_bboxes
    
    def get_class_mapping(self):
        return self.labels_to_names
        
    
    
        