from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import urllib
import zipfile
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import random

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
 
tf.app.flags.DEFINE_string('data_dir', '../../data/aerial', "Directory with training and testing images")
tf.app.flags.DEFINE_string('out_dir', '../../data/aerial_tfrecords', "Output directory to hold tfrecords")


def convert_to_tfrecord(dataset, output_file):
  """Converts a file to TFRecords."""
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for record in dataset:
      image = imread(record[0])
      if image.ndim < 3:
        image = gray2rgb(image)
      example = tf.train.Example(features=tf.train.Features(
        feature={
          'image': _bytes_feature(image.tobytes()),
          'label': _int64_feature(record[1])
        }))
      record_writer.write(example.SerializeToString())

                           
def main(argv=None):
  
  # Map class labels to integers
  class_to_label = {'Barren':0, 'Cultivated':1, 'Developed':2, 'Forest':3, 'Herbaceous':4, 'Shrub':5}

  # Create a list of training images with labels
  dataset = []
  for folder in os.listdir(os.path.join(FLAGS.data_dir, 'train')):
    for image in os.listdir(os.path.join(FLAGS.data_dir, 'train', folder)):
      dataset.append(
        (os.path.join(FLAGS.data_dir, 'train', folder, image),
        class_to_label[folder]))
         
  # Shuffle the list
  random.shuffle(dataset)
  filename = os.path.join(FLAGS.out_dir, 'aerial_train.tfrecords')
  
  if  os.path.exists(filename):
    print("File {0} already exists. Aborting ...".format(filename))
    return

  convert_to_tfrecord(dataset, filename) 

  # Create a list of training images with labels
  dataset = []
  for folder in os.listdir(os.path.join(FLAGS.data_dir, 'test')):
    for image in os.listdir(os.path.join(FLAGS.data_dir, 'test', folder)):
      dataset.append(
        (os.path.join(FLAGS.data_dir, 'test', folder, image),
        class_to_label[folder]))

  # Shuffle the list
  random.shuffle(dataset)
  filename = os.path.join(FLAGS.out_dir, 'aerial_validate.tfrecords')
   
  if  os.path.exists(filename):
    print("File {0} already exists. Aborting ...".format(filename))
    return

  convert_to_tfrecord(dataset, filename) 

  print('Done!')
 

if __name__ == '__main__':
  tf.app.run()
  
