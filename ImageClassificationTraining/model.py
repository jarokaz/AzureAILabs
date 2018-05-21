import tensorflow as tf
from tensorflow.python.keras.datasets import mnist, cifar10
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adadelta, Adam
from tensorflow.python.keras.callbacks import TensorBoard

from tensorflow.python.keras.applications.resnet50 import ResNet50 
from tensorflow.python.keras import models
from tensorflow.python.keras import layers


IMAGE_SHAPE = (224, 224, 3)
NUM_CLASSES = 6
INPUT_NAME = 'images'


def model_network(hidden_units):
  
    input = Input(shape=IMAGE_SHAPE, name=INPUT_NAME)
    conv_base = ResNet50(weights='imagenet',
                   include_top=False,
                   pooling = 'avg'
                   input_tensor=input)
    
    for layer in conv_base.layers:
        layer.trainable = False

    a = Flatten()(conv_base.output)
    a = Dense(hidden_units, activation='relu')(a)
    y = Dense(NUM_CLASSES, activation='softmax')(a)
    
    model = Model(inputs=input, outputs=y)
    
    return model
