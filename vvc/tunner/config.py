'''
'''
import os

from keras.applications.resnet50 import ResNet50

BASE_FOLDER = os.path.join(os.environ['HOME'], 'workspace/Maestria/')
KERAS_MODEL_FOLDER = os.environ['HOME']+'/.keras/models/'
MODELS_FOLDER = os.path.join(BASE_FOLDER, 'Model')
TRAIN_PATH = os.path.join(BASE_FOLDER, 'Videos/tf_pascal_voc')

model = ResNet50(weights='imagenet')
RESNET50_H5 = os.path.join(KERAS_MODEL_FOLDER, 'resnet50_weights_tf_dim_ordering_tf_kernels.h5')