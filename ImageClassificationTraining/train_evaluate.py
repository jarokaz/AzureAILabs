import os
import tensorflow as tf

from tensorflow.python.keras.optimizers import Adadelta, Adam
from tensorflow.python.keras.optimizers import Adadelta, Adam
from tensorflow.python.keras.estimator import model_to_estimator

import model


# Define input pipelines
def scale_image(image):

    """Scales image pixesl between -1 and 1"""
    image = image / 127.5
    image = image - 1.
    return image

  
def _parse(example_proto, augment):
  features = tf.parse_single_example(
        example_proto,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
  image = tf.decode_raw(features['image'], tf.uint8)
  image = tf.cast(image, tf.float32)
  image = scale_image(image)
  image = tf.reshape(image, model.IMAGE_SHAPE)
  
  #if augment:
  #  image = tf.image.random_flip_left_right(image)
     
  label = features['label']
  #label = tf.one_hot(label, NUM_CLASSES)
  return image, label



def input_fn(data_file, is_training, batch_size, num_parallel_calls, shuffle_buffer=5000):
 
  dataset = tf.data.TFRecordDataset(data_file)
  
  # Prefetch a batch at a time
  dataset = dataset.prefetch(buffer_size=batch_size)
    
  # Shuffle the records
  if is_training:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
  
  dataset = dataset.repeat(None if is_training else 1)
    
  # Parse records
  parse = lambda x: _parse(x, is_training)
  dataset = dataset.map(parse, num_parallel_calls=num_parallel_calls)
  
  # Batch, prefetch, and serve
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=1)
  
  iterator = dataset.make_one_shot_iterator()
  image_batch, label_batch = iterator.get_next()
  
  return {model.INPUT_NAME: image_batch}, label_batch


def serving_input_fn():
    input_image = tf.placeholder(shape=model.INPUT_SHAPE, dtype=tf.uint8)
    image = tf.cast(input_image, tf.float32)
    scaled_image = scale_image(image)
    
    return tf.estimator.export.ServingInputReceiver({model.INPUT_NAME: scaled_image}, {model.INPUT_NAME: input_image})



FLAGS = tf.app.flags.FLAGS

# Default global parameters
tf.app.flags.DEFINE_integer('batch_size', 32, "Number of images per batch")
tf.app.flags.DEFINE_integer('max_steps', 100000, "Number of steps to train")
tf.app.flags.DEFINE_string('job_dir', '../../jobdir/run1', "Checkpoints")
tf.app.flags.DEFINE_float('lr', 0.0005, 'Learning rate')
tf.app.flags.DEFINE_string('training_file', '../../data/aerial_tfrecords/aerial_train.tfrecords', "Training TFRecords file")
tf.app.flags.DEFINE_string('validation_file', '../../data/aerial_tfrecords/aerial_validate.tfrecords', "Validation TFRecords file")
tf.app.flags.DEFINE_string('verbosity', 'INFO', "Control logging level")
tf.app.flags.DEFINE_integer('num_parallel_calls', 12, 'Input parallelization')
tf.app.flags.DEFINE_integer('throttle_secs', 300, "Evaluate every n seconds")
tf.app.flags.DEFINE_integer('hidden_units', 256, "Hidden units in the FCNN layer")


def train_evaluate():

  #Create a keras model
  network_model = model.network(FLAGS.hidden_units)
  loss = 'sparse_categorical_crossentropy'
  metrics = ['accuracy']
  opt = Adadelta()
  network_model.compile(loss=loss, optimizer=opt, metrics=metrics)

  #Convert the the keras model to tf estimator
  estimator = model_to_estimator(keras_model = network_model, model_dir=FLAGS.job_dir)
    
  #Create training, evaluation, and serving input functions
  train_input_fn = lambda: input_fn(data_file=FLAGS.training_file, 
                                    is_training=True, 
                                    batch_size=FLAGS.batch_size, 
                                    num_parallel_calls=FLAGS.num_parallel_calls)
    
  valid_input_fn = lambda: input_fn(data_file=FLAGS.training_file, 
                                    is_training=False, 
                                    batch_size=FLAGS.batch_size, 
                                    num_parallel_calls=FLAGS.num_parallel_calls)
  
  #Create training and validation specifications
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, 
                                      max_steps=FLAGS.max_steps)
  
  export_latest = tf.estimator.FinalExporter("image_classifier", serving_input_fn)
    
  eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, 
                                    steps=None,
                                    throttle_secs=FLAGS.throttle_secs,
                                    exporters=export_latest)
  
 
  #Start training
  tf.logging.set_verbosity(FLAGS.verbosity)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  

def main(argv=None):
 
  if tf.gfile.Exists(FLAGS.job_dir):
    tf.gfile.DeleteRecursively(FLAGS.job_dir)
  tf.gfile.MakeDirs(FLAGS.job_dir)
  
  train_evaluate()
  

if __name__ == '__main__':
  tf.app.run()
  
