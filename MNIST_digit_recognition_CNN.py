# identifying digits from the MNIST dataset using CNN

import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.99:
            print('\nReached 99% accuracy, cancelling training !')
            self.model.stop_training = True

callback = myCallback()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train/255.0, x_test/255.0

# Reshape the data for input
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Define Model
model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the Model
model_fit = model.fit(x_train, y_train, epochs=30, callbacks=callback)


