import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # this should supress TensorFlow outputs
import tensorflow as tf
import time   # to measure performance

# define some functions to preprocess the data
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label = parts[-2]
    label = tf.strings.to_number(label, tf.int32)
    #label2 = train_labels.loc[label.decode('utf-8'), 'individual_id']

    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [1080, 1920])
    return image, label

# helper function for displaying image
def show(image, label):
    plt.figure()
    plt.imshow(image)
    label = label.numpy().decode('utf-8')
    #label = train_labels.loc[label, 'individual_id']
    plt.title(label)
    plt.axis('off')
# create a TensorFlow Input Pipeline


strt_time = time.time()
print("Grabbing files")
train_files = tf.data.Dataset.list_files("/home/brandon/kaggle/data2/*/*")
end_time = time.time() - strt_time
print("Done grabbing files. Took (sec):", end_time, "\n\n")

print("parsing training data")
strt_time = time.time()
train_data = train_files.map(parse_image)
train_data = train_data.batch(8)
end_time = time.time() - strt_time
print("done parsing training data")
print("Took (sec):", end_time, "\n\n")

# create a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, (100,100), strides=(100,100), input_shape=(1080, 1920, 1)),
    tf.keras.layers.Conv2D(20, (5, 5), strides=(5,5)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(15587)
    ])

#  defijne a loss
# define an optimizer
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
    )
model.fit(train_data, epochs=10)