'''

'''
import os

base_folder = os.path.join(os.environ['HOME'], 'workspace/Maestria')
video_folder = os.path.join(base_folder, 'Videos', 'Original')
output_folder = os.path.join(base_folder, 'Videos', 'vvc')

model_folder = os.path.join(base_folder, 'Model')

models = ['frcnn-resnet50', 'frcnn-resnet50-transfer',
          'YOLOv3', 'YOLOv3-transfer', 'YOLOv3-tiny', 'YOLOv3-tiny-transfer',
          'vvc1-yolov3', 'vvc2-yolov3']