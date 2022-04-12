from glob import glob
import os
import random
from tqdm import tqdm
import tensorflow as tf
import pdb

def serialize_example(image, label):

    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def make_tfrecords(path, record_file='subjects.tfrecords'):
  classes = os.listdir(path)
  with tf.io.TFRecordWriter(record_file) as writer:
    files_list = glob(path + '/*/*')
    random.shuffle(files_list)
    for filename in tqdm(files_list):
      image = tf.io.read_file(filename)
      image = tf.image.decode_jpeg(image, channels=3)
      image = tf.image.resize(image, [256, 256])
      image_string = tf.io.encode_jpeg(tf.cast(image, tf.uint8), format='rgb').numpy()
      #image_string = open(filename, 'rb').read()
      category = filename.split('\\')[-2]
      label = classes.index(category)
      tf_example = serialize_example(image_string, label)
      writer.write(tf_example)

make_tfrecords("C:\\Users\\DNNeu\\kaggle\\HappyWhale\\Data_Small\\")