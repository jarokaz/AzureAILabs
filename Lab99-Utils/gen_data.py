import numpy as np
import pandas as pd
import tensorflow as tf
import os
import PIL

FLAGS = tf.app.flags.FLAGS

# Default global parameters
tf.app.flags.DEFINE_string('image_dir', '../../data/wood/unprocessed/IMAGES', 'Directory with unprocessed images')
tf.app.flags.DEFINE_string('label_file_path', '../../data/wood/unprocessed/IMAGES/manlabel.txt', 'Path to the label file')
tf.app.flags.DEFINE_string('output_dir', '../../data/wood/tfrecords', 'Output directory for generated tfrecords files')


def gen_samples(image, group):
    for _, row in group.iterrows():
        print(row['image'])

def gen_images(label_file_path, image_dir, output_dir):
    index = pd.read_csv(label_file_path, delim_whitespace=True, header=None, names=['image', 'min_y', 'min_x', 'max_y', 'max_x', 'label'])
    assert len(index) != 0

    gb = index.groupby(index['image'])
    for name, group in gb:
       image = PIL.Image.open(os.path.join(FLAGS.image_dir, name))
       gen_samples(image, group)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.DeleteRecursively(FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    print("Generating images")
    gen_images(FLAGS.label_file_path, FLAGS.image_dir, FLAGS.output_dir)

if __name__ == '__main__':
    tf.app.run()



