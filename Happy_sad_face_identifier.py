import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

data_path = 'C:\\Users\\navin\\Desktop\\PyCharm Projects\\TF projects\\Data\\happy-or-sad.zip'
# path = f"{getcwd()}/../tmp2/happy-or-sad.zip"

# Read data from the zip file
zip_ref = zipfile.ZipFile(data_path, 'r')

# Create a tmp folder and place the extracted data in there
zip_ref.extractall('/tmp/h-o-s')
zip_ref.close()

# Define callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.99:
            print(f'accuracy more than 99% terminating model training')
            self.model.stop_training = True

# Define instance of callback class
callback = myCallback()

# Define model structure
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr = 0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Define image data loader to pass images to model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory('/tmp/h-o-s/',
                                                    target_size=(150, 150),
                                                    batch_size=10,
                                                    class_mode='binary')
# Train the model
model.fit(train_generator,
          steps_per_epoch=8,
          epochs=50,
          verbose=1,
          callbacks=[callback])


