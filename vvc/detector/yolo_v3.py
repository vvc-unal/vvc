'''
'''
import os

from keras import backend as K
import numpy as np
from PIL import Image
from yolo import YOLO
from yolo3.utils import letterbox_image

from vvc.config import model_folder


class YOLOV3(object):
    '''
    classdocs
    '''

    def __init__(self, model_name):
        '''
        Constructor
        '''
        self.model_name = model_name
        config = {
             "model_path": os.path.join(model_folder, model_name, 'yolo.h5'),
             "anchors_path": os.path.join(model_folder, model_name, 'anchors.txt'),
             "classes_path": os.path.join(model_folder, model_name, 'classes.txt')
            }
        self.yolo = YOLO(**config)
        
    def predict(self, frame):
        
        image = Image.fromarray(frame)
        
        if self.yolo.model_image_size != (None, None):
            assert self.yolo.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.yolo.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.yolo.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.yolo.sess.run(
            [self.yolo.boxes, self.yolo.scores, self.yolo.classes],
            feed_dict={
                self.yolo.yolo_model.input: image_data,
                self.yolo.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        
        final_bboxes = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.yolo.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            
            bbox = {}
            bbox['class'] = predicted_class
            bbox['box'] = [left, top, right, bottom]
            bbox['prob'] = score
            final_bboxes.append(bbox)
 
        return final_bboxes
    
    def get_class_mapping(self):
        return {k: v for k, v in enumerate(self.yolo.class_names)}
        
    
    
        