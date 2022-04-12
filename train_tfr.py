from glob import glob
import random
import os
import tensorflow as tf
import tensorflow_addons as tfa
from InceptionNet import InceptionNet
import pdb

def _parse_image_function(example):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_jpeg(features['image'], channels=3)
    #image = tf.image.resize(image, [28, 28])
    #image = tf.image.resize(image, [256, 256])

    label = tf.cast(features['label'], tf.int32)

    return image, label


def read_dataset(filename, batch_size):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(500)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    #dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

ds_train = read_dataset('subjects.tfrecords', 200)


# create a simple neural network model
'''model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, (10,10), strides=(5,5), input_shape=(256, 256, 3), activation='relu'),
    tf.keras.layers.Conv2D(20, (3, 3), strides=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(10, (2, 2), strides=(1,1), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(528, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation=None),
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])

'''

# create an inception based model
model = InceptionNet().assemble_full_model()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

#  defijne a loss
# define an optimizer
model.compile(optimizer=opt,
    loss={'output0' : tfa.losses.TripletSemiHardLoss(),
    'output1' : tfa.losses.TripletSemiHardLoss(),
    'output2' : tfa.losses.TripletSemiHardLoss()},
    loss_weights = {'output0': 0.2, 'output1' : 0.2, 'output2' : 1.0},
    metrics={'output2' : 'accuracy'}
)

hist = model.fit(ds_train, epochs=100)
training_loss = hist.history['loss']
training_acc = hist.history['output2_accuracy']

import matplotlib.pyplot as plt
plt.figure()
plt.plot(training_loss)
plt.title("Training Loss")

plt.figure()
plt.plot(training_acc)
plt.title("Training Accuracy")

plt.show()