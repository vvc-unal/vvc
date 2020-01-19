from datetime import datetime, timedelta
import itertools
import operator
import os

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from vvc.utils import json_utils
from vvc import config as vvc_config
from vvc.tracker.iou_tracker import IOUTracker
from vvc.video_data import VideoData
from vvc.video.skvideo_writer import SKVideoWriter
from vvc.video.video_wrapper import VideoWrapper


# keras
import keras

import tensorflow as tf
import logging


def get_session():
	''' Set tf backend to allow memory to grow, instead of claiming everything '''
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


def miliseconds_from(last_time):
	now = datetime.now()
	diff = (now - last_time) / timedelta(microseconds=1) / 1000
	return diff, now


class VVC(object):

	def __init__(self, detector, tracker=IOUTracker()):
		self.model_name = detector.model_name
		self.obj_detector = detector
		self.tracker = tracker

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

		return img, f

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
		text_label = label

		(retval, base_line) = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_DUPLEX , 0.5, 1)
		text_org = (x1, y1)

		base_line += 1
		point1 = (text_org[0] - 5, text_org[1] + base_line - 5)
		point2 = (text_org[0] + retval[0] + 5, text_org[1] - retval[1] - 5)

		cv2.rectangle(img_box, point1, point2, (0, 0, 0), 2)
		cv2.rectangle(img_box, point1, point2, (255, 255, 255), -1)
		cv2.putText(img_box, text_label, text_org, cv2.FONT_HERSHEY_DUPLEX , 0.5, (0, 0, 0), 1)

		return img_box

	def count(self, video_name,
			  frame_rate_factor=1,
			  filter_tags=['bicycle', 'car', 'motorbike', 'bus', 'truck'],
			  show_obj_id=True):
		# Load video and parameters

		logging.info('Load video and parameters')
		input_video_file = os.path.join(vvc_config.video_folder, video_name)

		video_name_no_suffix = Path(input_video_file).stem

		output_folder = os.path.join(vvc_config.output_folder, video_name_no_suffix)
		output_video_file = os.path.join(output_folder, self.model_name + '.mp4')

		input_video = VideoWrapper(input_video_file)

		frame_rate = input_video.avg_frame_rate()

		frame_rate = frame_rate * frame_rate_factor

		logging.info("Counting ...")

		# Init process

		data = VideoData()
		data.timestamps['start'] = datetime.now().isoformat()
		data.video.input_file = input_video_file
		data.video.output_file = output_video_file

		reader = input_video.video_reader()

		video_writer = SKVideoWriter(output_video_file, frame_rate)

		frame_id = 0
		total_frames = input_video.total_frames()

		class_mapping = self.obj_detector.get_class_mapping()

		class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3).tolist() for v in class_mapping}

		data.timestamps['warm'] = datetime.now().isoformat()

		last_time = datetime.now()

		with tqdm(total=total_frames, unit='frame') as pbar:

			for frame in reader:

				frame_id += 1
				img_name = str(frame_id) + ".jpg"

				tqdm_step = int(total_frames / 10)
				if frame_id % tqdm_step == 0:
					pbar.update(tqdm_step)

				# Save frame data
				frame_data = data.add_frame_data()
				frame_data.name = img_name
				frame_data.timestamps['read'], last_time = miliseconds_from(last_time)

				# Image preprocesing
				img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

				# Resize input image
				X, scale_factor = self.format_img_yolo(img, 600)

				img_scaled = X.astype(np.uint8)

				frame_data.timestamps['preprocessing'], last_time = miliseconds_from(last_time)

				# Objetct detection

				bboxes = self.obj_detector.predict(X)

				for bbox in bboxes:

					tag = bbox['class']

					# Save annotations
					if tag in filter_tags:
						object_data = frame_data.add_object()
						object_data.tag = tag
						object_data.box = bbox['box']
						object_data.probability = float(bbox['prob'])

				frame_data.timestamps['detection'], last_time = miliseconds_from(last_time)

				# Tracking the objects
				tracked_objects = self.tracker.tracking(img_scaled, frame_data.objects)

				frame_data.timestamps['tracking'], last_time = miliseconds_from(last_time)

				# Plot tracking results
				img_tracks = img_scaled.copy()

				for object_data in tracked_objects:

					box = object_data.boxes[-1]
					color = class_to_color[object_data.tag]
					label = '{}'.format(object_data.name)

					track_data = frame_data.add_track()
					track_data.id = object_data.name
					track_data.box = np.array(box) * (1/scale_factor)

					if not show_obj_id:
						label = label.split(sep=' ')[0]

						label += ' {0:.0%}'.format(object_data.probability)

					img_tracks = self.plot_box(img_tracks, box, color, label)

				# Save final image
				img_post = img_tracks

				img_post = cv2.cvtColor(img_post, cv2.COLOR_RGB2BGR)

				video_writer.writeFrame(img_post)

				frame_data.timestamps['postprocessing'], last_time = miliseconds_from(last_time)

			reader.close()

		data.timestamps['end_loop'] = datetime.now().isoformat()

		video_writer.close()

		data.timestamps['write'] = datetime.now().isoformat()

		# Save data to json file
		json_file = output_video_file + ".json"
		json_utils.save_to_json(data, json_file)

		return json_file


