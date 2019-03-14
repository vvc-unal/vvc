'''

'''
import os
import pickle
import sys

from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import keras_frcnn.resnet as nn
import numpy as np

from vvc import config as vvc_config

class FasterRCNN(object):
    
    def __init__(self, model_name, num_rois = 32):
        sys.setrecursionlimit(40000)
        config_output_filename = os.path.join(vvc_config.model_folder, model_name, 'config.pickle')
    
        with open(config_output_filename, 'rb') as f_in:
            self.C = pickle.load(f_in)
    
        # turn off any data augmentation at test time
        self.C.use_horizontal_flips = False
        self.C.use_vertical_flips = False
        self.C.rot_90 = False
        self.class_mapping = self.C.class_mapping
        
        # Overwrite model path
        self.C.model_path = os.path.join(vvc_config.model_folder, self.model_name, 'model.hdf5')
    
        if 'bg' not in self.class_mapping:
            self.class_mapping['bg'] = len(self.class_mapping)
    
        self.class_mapping = {v: k for k, v in self.class_mapping.items()}
        print(self.class_mapping)
        self.C.num_rois = num_rois
    
        if K.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
            input_shape_features = (1024, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, 1024)
    
        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)
    
        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)
    
        # define the RPN, built on the base layers
        num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)
    
        classifier = nn.classifier(feature_map_input, roi_input, self.C.num_rois, nb_classes=len(self.class_mapping), trainable=True)
    
        self.model_rpn = Model(img_input, rpn_layers)
        self.model_classifier_only = Model([feature_map_input, roi_input], classifier)
    
        self.model_classifier = Model([feature_map_input, roi_input], classifier)
    
        self.model_rpn.load_weights(self.C.model_path, by_name=True)
        self.model_classifier.load_weights(self.C.model_path, by_name=True)
    
        self.model_rpn.compile(optimizer='sgd', loss='mse')
        self.model_classifier.compile(optimizer='sgd', loss='mse')
    
        self.bbox_threshold = 0.8
        
    def predict(self, X):
        if K.image_dim_ordering() == 'tf':
                X = np.transpose(X, (0, 2, 3, 1))
    
        # get the feature maps and output from the RPN
        [Y1, Y2, F] = self.model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // self.C.num_rois + 1):
            ROIs = np.expand_dims(R[self.C.num_rois * jk:self.C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // self.C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < self.bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= self.C.classifier_regr_std[0]
                    ty /= self.C.classifier_regr_std[1]
                    tw /= self.C.classifier_regr_std[2]
                    th /= self.C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
        
        final_bboxes = []    
        for key in bboxes:
            bbox = np.array(bboxes[key])
            
            # Get the boxes with higher probabilities
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            
            for jk in range(new_boxes.shape[0]):                
                # Save annotations
                box = {}
                box['class'] = key
                box['box'] = new_boxes[jk, :]
                box['prob'] = new_probs[jk]
                final_bboxes.append(box)
                    
        return final_bboxes