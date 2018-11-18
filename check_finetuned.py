"""Check whether the BERT variable is changed."""

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  'raw_ckpt', None, 'Pre-trained BERT model path.')
flags.DEFINE_string(
  'trained_ckpt', None, 'Trained model path.')

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  # fine-tune flag
  flag = False
  # get varibles key and value in pre-trained model
  raw_reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.raw_ckpt)
  raw = raw_reader.get_variable_to_shape_map()
  for key in raw.keys():
    if 'bert' in key:
      continue
    raw.pop(key)
  # get varibales in trained model
  reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.trained_ckpt)
  trained = reader.get_variable_to_shape_map()
  for key in trained:
    if 'bert' in key and 'adam' not in key:
      if not np.allclose(reader.get_tensor(key), raw_reader.get_tensor(key)):
        flag = True
        break
      raw.pop(key)
  # if variables loss
  if len(raw):
    flag = True
  if flag:
    tf.logging.info(' The model is finetuned.')
  else:
    tf.logging.info(' The model is not finetuned.')


if __name__ == '__main__':
  flags.mark_flag_as_required("raw_ckpt")
  flags.mark_flag_as_required("trained_ckpt")
  tf.app.run()


