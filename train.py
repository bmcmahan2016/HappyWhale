import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # this should supress TensorFlow outputs
import tensorflow as tf
import time   # to measure performance
import pdb

# define some functions to preprocess the data
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.


ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/DNNeu/kaggle/HappyWhale/Data2/',
    labels='inferred',
    label_mode = "int",
    color_mode = 'rgb',
    batch_size = 32,
    image_size=(256, 256),
    shuffle=True,
    seed=12345,
    validation_split=0.1,
    subset="training",
)



# create a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, (10,10), strides=(10,10), input_shape=(256, 256, 3)),
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
model.fit(ds_train, epochs=10)