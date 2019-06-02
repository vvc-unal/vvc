'''

'''
import os

base_folder = os.path.join(os.environ['HOME'], 'workspace/Maestria')
video_folder = os.path.join(base_folder, 'Videos', 'Train')
output_folder = os.path.join(base_folder, 'Videos', 'vvc')

model_folder = os.path.join(base_folder, 'Model')

models = ['frcnn-resnet50', 'frcnn-resnet50-transfer', 'frcnn-resnet50-tunned',
          'YOLOv3', 'YOLOv3-transfer']