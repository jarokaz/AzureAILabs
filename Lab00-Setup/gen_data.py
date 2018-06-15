import numpy as np
import pandas as pd
import tensorflow as tf
import os
import PIL

FLAGS = tf.app.flags.FLAGS

# Default global parameters
tf.app.flags.DEFINE_string('input_dir', '../../data/wood/unprocessed/IMAGES', 'Directory with unprocessed images')
tf.app.flags.DEFINE_string('label_file_path', '../../data/wood/unprocessed/IMAGES/manlabel.txt', 'Path to the label file')
tf.app.flags.DEFINE_string('output_tfrecords', '../../data/wood/tfrecords', 'Output directory for generated tfrecords files')
tf.app.flags.DEFINE_string('output_images', '../../data/wood/processed', 'Output directory for processed')

def process_grid(image, group, output_dir):
    for i, row in group.iterrows():
        im = image.crop((row['min_x'], row['min_y'], row['max_x'], row['max_y']))
        filename = row['image'] + str(i) + "-" + row['label'] + ".jpg"
        filename = os.path.join(output_dir, filename)
        im.save(filename)

def gen_images(label_file_path, image_dir, output_dir):
    index = pd.read_csv(label_file_path, delim_whitespace=True, header=None, names=['image', 'min_y', 'min_x', 'max_y', 'max_x', 'label'])
    assert len(index) != 0

    gb = index.groupby(index['image'])
    for name, group in gb:
       image = PIL.Image.open(os.path.join(FLAGS.input_dir, name))
       process_grid(image, group, output_dir)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.output_images):
        tf.gfile.DeleteRecursively(FLAGS.output_images)
    if tf.gfile.Exists(FLAGS.output_tfrecords):
        tf.gfile.DeleteRecursively(FLAGS.output_tfrecords)
    tf.gfile.MakeDirs(FLAGS.output_images)
    tf.gfile.MakeDirs(FLAGS.output_tfrecords)
    print("Generating images")
    gen_images(FLAGS.label_file_path, FLAGS.input_dir, FLAGS.output_images)

if __name__ == '__main__':
    tf.app.run()



