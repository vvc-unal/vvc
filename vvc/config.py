'''

'''
import os

base_folder = os.path.join(os.environ['HOME'], 'workspace/Maestria')
video_folder = os.path.join(base_folder, 'Videos', 'original')

model_folder = os.path.join(base_folder, 'Model')

models = ['frcnn-resnet50', 'frcnn-resnet50-transfer']

videoName = 'CL 53 X CRA 60 910-911.mp4'