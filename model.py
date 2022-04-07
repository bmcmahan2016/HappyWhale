import tensorflow as tf


model = tf.keras.Sequential([
	tf.keras.layers.Conv2D(10, (100,100), strides=(100,100), input_shape=(1080, 1920, 3)),
	tf.keras.layers.Conv2D(20, (5, 5), strides=(5,5)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(15000)
	])
print(model.summary())