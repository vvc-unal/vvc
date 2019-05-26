from datetime import datetime
import itertools
import operator
import os
import shutil
import sys

import cv2
import json
import numpy as np
from pathlib import Path

from vvc import video_utils, json_utils
from vvc import config as vvc_config
from vvc.tracker import NaiveTracker
from vvc.video_data import VideoData

class VVC(object):
	
	def __init__(self, detector, tracker=NaiveTracker()):
		self.model_name = detector.model_name
		self.obj_detector = detector
		self.tracker = tracker
		

	def cleanup(self):
		print("cleaning up...")
		shutil.rmtree(self.output_img, ignore_errors=True)
		
		os.makedirs(self.output_img, exist_ok=True)
	
	def format_img_yolo(self, img, img_min_side):
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
	
	def main(self, filter_tags, show_obj_id):
		
		data = VideoData()
		data.video.input_file = self.input_video_file
		data.video.output_file = self.output_video_file
		
		# vott file
		vott_file_path = Path(self.input_video_file + '.json')
		if vott_file_path.exists():
			with vott_file_path.open() as json_data:
				vott = json.load(json_data)
								
		print("anotating ...")
	
		reader = video_utils.video_reader(self.input_video_file)
	
		frame_id = 0
		
		class_mapping = self.obj_detector.get_class_mapping()
			
		class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3).tolist() for v in class_mapping}
	
		for frame in reader:
			
			frame_id += 1
			img_name = str(frame_id) + ".jpg"
			
			# Save frame data
			frame_data = data.add_frame_data()
			frame_data.name = img_name
			frame_data.timestamps['start'] = datetime.now().isoformat()
			
			# Image preprocesing
			img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			
			# Plot ignore box
			try:
				vott_frame = vott['frames'][str(frame_id)]
				ignore_tags = list(filter(lambda x: 'ignore' in x['tags'], vott_frame))
				if len(ignore_tags) > 0:
					box = ignore_tags[0]['box']
					ignore_box = np.array([box['x1'], box['y1'], box['x2'], box['y2']]).astype(int)
					(x1, y1, x2, y2) = ignore_box
					color = [0, 0, 0]
					cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
			except:
				pass
			
			# Resize image
			X = self.format_img_yolo(img, 600)
	
			if False: # Faster R-CNN
				img_scaled = np.transpose(X.copy()[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
				img_scaled[:, :, 0] += 123.68
				img_scaled[:, :, 1] += 116.779
				img_scaled[:, :, 2] += 103.939
	
				img_scaled = img_scaled.astype(np.uint8)
			else:
				img_scaled = X.astype(np.uint8)
			
			frame_data.timestamps['preprocessing_end'] = datetime.now().isoformat()
			
			bboxes = self.obj_detector.predict(X)
					
			for bbox in bboxes:
				
				tag = bbox['class']
				
				# Save annotations
				if tag in filter_tags:
					object_data = frame_data.add_object()
					object_data.tag = tag
					object_data.box = bbox['box']
					object_data.probability = bbox['prob']
			
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
				
				if not show_obj_id:
					label = label.split(sep=' ')[0]
				
				img_tracks = self.plot_box(img_tracks, box, color, label)
			
			# Save final image
			img_scaled = img_tracks
			cv2.imwrite(os.path.join(self.output_img, '{:05d}'.format(frame_id) + ".jpg"), img_scaled)
			
			frame_data.timestamps['postprocessing_end'] = datetime.now().isoformat()
			
		# Save data to json file
		json_file = self.output_video_file + ".json"
		json_utils.save_to_json(data, json_file)
		
	def count(self, video_name, frame_rate_factor=0.1, 
			filter_tags=['bicycle', 'person', 'car', 'motorbike', 'bus', 'truck'], 
			show_obj_id=True):
		
		self.video_name = video_name
		self.input_video_file = os.path.join(vvc_config.video_folder, video_name)
		
		video_name_no_suffix = Path(self.input_video_file).stem
		
		self.output_folder = os.path.join(vvc_config.output_folder, video_name_no_suffix)
		self.output_video_file = os.path.join(self.output_folder, self.model_name + '.mp4')
		self.output_img = os.path.join(self.output_folder, self.model_name)
		
		self.cleanup()
		
		frame_rate = video_utils.get_avg_frame_rate(self.input_video_file)
		
		frame_rate = frame_rate * frame_rate_factor
		
		print("Main ...")
		self.main(filter_tags=filter_tags, show_obj_id=show_obj_id)
		
		print("saving to video ..")
		video_utils.save_to_video(self.output_img, self.output_video_file, frame_rate)
		
		print("Done..")
		
			
