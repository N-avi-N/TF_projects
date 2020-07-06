# Creating simple NN in tensorflow to predict house prices

import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
y = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

model.fit(x, y, epochs=10)

print('sample data prediction is ', model.predict([4.5]))