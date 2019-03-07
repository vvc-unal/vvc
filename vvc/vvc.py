import itertools
import operator
import os
import pickle
import sys
from datetime import datetime
import shutil

import cv2
from keras import backend as K
from keras.layers import Input
from keras.models import Model

from keras_frcnn import roi_helpers
import keras_frcnn.resnet as nn
import numpy as np

from vvc import video_utils, json_utils
from vvc import config as vvc_config
from vvc.tracker import NaiveTracker
from vvc.video_data import VideoData

num_rois = 32
frame_rate = 30

class VVC(object):
	
	def __init__(self, video_name, model_name):
		self.model_name = model_name
		self.tracker = NaiveTracker()
		self.video_name = video_name

		self.input_video_file = os.path.join(vvc_config.video_folder, video_name)
		self.output_folder = os.path.join(vvc_config.base_folder, 'Videos', model_name)
		self.output_video_file = os.path.join(self.output_folder, video_name)
		self.output_path = os.path.join(self.output_folder, 'tmp_output')
		

	def cleanup(self):
		print("cleaning up...")
		shutil.rmtree(self.output_path, ignore_errors=True)
		
		os.makedirs(self.output_path, exist_ok=True)
	
	def format_img(self, img, C):
		img_min_side = float(C.im_size)
		(height, width, _) = img.shape
	
		if width <= height:
			f = img_min_side / width
			new_height = int(f * height)
			new_width = int(img_min_side)
		else:
			f = img_min_side / height
			new_width = int(f * width)
			new_height = int(img_min_side)
		img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
		img = img[:, :, (2, 1, 0)]
		img = img.astype(np.float32)
		img[:, :, 0] -= C.img_channel_mean[0]
		img[:, :, 1] -= C.img_channel_mean[1]
		img[:, :, 2] -= C.img_channel_mean[2]
		img /= C.img_scaling_factor
		img = np.transpose(img, (2, 0, 1))
		img = np.expand_dims(img, axis=0)
		return img
	
	def accumulate(self, l):
		it = itertools.groupby(l, operator.itemgetter(0))
		for key, subiter in it:
			yield key, sum(item[1] for item in subiter)
	
	def plot_box(self, img, box, color, label):
		'''
		Plot a box on an image
		'''
		img_box = img.copy()
		(x1, y1, x2, y2) = box
				
		cv2.rectangle(img_box, (x1, y1), (x2, y2), color, 2)
		textLabel = label
				
		(retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_DUPLEX , 0.5, 1)
		textOrg = (x1, y1)
	
		baseLine += 1
		point1 = (textOrg[0] - 5, textOrg[1] + baseLine - 5)
		point2 = (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5)
	
		cv2.rectangle(img_box, point1, point2, (0, 0, 0), 2)
		cv2.rectangle(img_box, point1, point2, (255, 255, 255), -1)
		cv2.putText(img_box, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX , 0.5, (0, 0, 0), 1)
		
		return img_box
	
	def main(self):
		
		data = VideoData()
		data.video.input_file = self.input_video_file
		data.video.output_file = self.output_video_file
		
		sys.setrecursionlimit(40000)
		config_output_filename = os.path.join(vvc_config.model_folder, self.model_name, 'config.pickle')
	
		with open(config_output_filename, 'rb') as f_in:
			C = pickle.load(f_in)
	
		# turn off any data augmentation at test time
		C.use_horizontal_flips = False
		C.use_vertical_flips = False
		C.rot_90 = False
		class_mapping = C.class_mapping
		
		# Overwrite model path
		C.model_path = os.path.join(vvc_config.model_folder, self.model_name, 'model.hdf5')
	
		if 'bg' not in class_mapping:
			class_mapping['bg'] = len(class_mapping)
	
		class_mapping = {v: k for k, v in class_mapping.items()}
		print(class_mapping)
		class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3).tolist() for v in class_mapping}
		C.num_rois = num_rois
	
		if K.image_dim_ordering() == 'th':
			input_shape_img = (3, None, None)
			input_shape_features = (1024, None, None)
		else:
			input_shape_img = (None, None, 3)
			input_shape_features = (None, None, 1024)
	
		img_input = Input(shape=input_shape_img)
		roi_input = Input(shape=(C.num_rois, 4))
		feature_map_input = Input(shape=input_shape_features)
	
		# define the base network (resnet here, can be VGG, Inception, etc)
		shared_layers = nn.nn_base(img_input, trainable=True)
	
		# define the RPN, built on the base layers
		num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
		rpn_layers = nn.rpn(shared_layers, num_anchors)
	
		classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)
	
		model_rpn = Model(img_input, rpn_layers)
		model_classifier_only = Model([feature_map_input, roi_input], classifier)
	
		model_classifier = Model([feature_map_input, roi_input], classifier)
	
		model_rpn.load_weights(C.model_path, by_name=True)
		model_classifier.load_weights(C.model_path, by_name=True)
	
		model_rpn.compile(optimizer='sgd', loss='mse')
		model_classifier.compile(optimizer='sgd', loss='mse')
	
		bbox_threshold = 0.8
		
		print("anotating ...")
	
		reader = video_utils.video_reader(self.input_video_file)
	
		frame_id = -1
	
		for frame in reader:
			
			frame_id += 1
			img_name = str(frame_id) + ".jpg"
			
			# Save frame data
			frame_data = data.add_frame_data()
			frame_data.name = img_name
			frame_data.timestamps['start'] = datetime.now().isoformat()
			
			img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			X = self.format_img(img, C)
	
			img_scaled = np.transpose(X.copy()[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
			img_scaled[:, :, 0] += 123.68
			img_scaled[:, :, 1] += 116.779
			img_scaled[:, :, 2] += 103.939
	
			img_scaled = img_scaled.astype(np.uint8)
			
			frame_data.timestamps['preprocessing_end'] = datetime.now().isoformat()
			
			if K.image_dim_ordering() == 'tf':
				X = np.transpose(X, (0, 2, 3, 1))
	
			# get the feature maps and output from the RPN
			[Y1, Y2, F] = model_rpn.predict(X)
	
			R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
	
			# convert from (x1,y1,x2,y2) to (x,y,w,h)
			R[:, 2] -= R[:, 0]
			R[:, 3] -= R[:, 1]
	
			# apply the spatial pyramid pooling to the proposed regions
			bboxes = {}
			probs = {}
	
			for jk in range(R.shape[0] // C.num_rois + 1):
				ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
				if ROIs.shape[1] == 0:
					break
	
				if jk == R.shape[0] // C.num_rois:
					# pad R
					curr_shape = ROIs.shape
					target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
					ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
					ROIs_padded[:, :curr_shape[1], :] = ROIs
					ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
					ROIs = ROIs_padded
	
				[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
	
				for ii in range(P_cls.shape[1]):
	
					if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
						continue
	
					cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
	
					if cls_name not in bboxes:
						bboxes[cls_name] = []
						probs[cls_name] = []
	
					(x, y, w, h) = ROIs[0, ii, :]
	
					cls_num = np.argmax(P_cls[0, ii, :])
					try:
						(tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
						tx /= C.classifier_regr_std[0]
						ty /= C.classifier_regr_std[1]
						tw /= C.classifier_regr_std[2]
						th /= C.classifier_regr_std[3]
						x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
					except:
						pass
					bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
					probs[cls_name].append(np.max(P_cls[0, ii, :]))
					
			for key in bboxes:
				bbox = np.array(bboxes[key])
				
				# Get the boxes with higher probabilities
				new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
				
				for jk in range(new_boxes.shape[0]):				
					# Save annotations
					object_data = frame_data.add_object()
					object_data.tag = key
					object_data.box = new_boxes[jk, :]
					object_data.probability = new_probs[jk]
			
			frame_data.timestamps['detection_end'] = datetime.now().isoformat()
			
			# Tracking the objects
			tracked_objects = self.tracker.tracking(frame_data.objects)
			
			frame_data.timestamps['tracking_end'] = datetime.now().isoformat()
			
			# Plot tracking results
			img_tracks = img_scaled.copy()
			
			for object_data in tracked_objects:
				
				box = object_data.box
				color = class_to_color[object_data.tag]
				label = '{}'.format(object_data.name)
				
				img_tracks = self.plot_box(img_tracks, box, color, label)

			
			# Save final image
			img_scaled = img_tracks
			cv2.imwrite(os.path.join(self.output_path, '{:05d}'.format(frame_id) + ".jpg"), img_scaled)
			
			frame_data.timestamps['postprocessing_end'] = datetime.now().isoformat()
			
		# Save data to json file
		json_file = os.path.join(self.output_folder, self.video_name + ".json")
		json_utils.save_to_json(data, json_file)
		
	def count(self):
		self.cleanup()
		
		frame_rate = video_utils.get_avg_frame_rate(self.input_video_file)
		
		print("Main ...")
		self.main()
		
		print("saving to video ..")
		video_utils.save_to_video(self.output_path, self.output_video_file, frame_rate)
		
		print("Done..")
