import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import urllib.request
from tensorflow.keras.optimizers import RMSprop

# Download and save weights h5 file for inception v3 network
url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 '
filename = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
data_path = 'C:\\Users\\navin\\Desktop\\PyCharm Projects\\TF projects\\Data\\Incepton_v3_weights'
fullfilename = os.path.join(data_path, filename)
urllib.request.urlretrieve(url, fullfilename)

# import the inception v3 model
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,      # this removes the classification dense layers at the end of a network
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers of the pre trained model non trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

# Print Model summary
pre_trained_model.summary()

# make layer 7 as the lsat layer
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer/mixed7 output shape : ', last_layer.output_shape)
last_output = last_layer.output

# Define a callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.999:
            print('Reached 99.9% accuracy, terminating training !')
            self.model.stop_training = True

callback = myCallback()

# Taking output from last layer and adding final prediction layers to it

# Flatten the output from last layer
x = layers.Flatten()(last_output)

# Add fully connected layer with 1024 hidden units
x = layers.Dense(1024, activation='relu')(x)

# Add dropout to the layer
x = layers.Dropout(0.2)(x)

# Add the last layer fpr human or horse classification
x = layers.Dense(1, activation='sigmoid')(x)

# Take the classification portion of the pre-trained layer and then add Add the new layers at the end of it
model = Model(pre_trained_model.input, x)
print(pre_trained_model.input,)
# Compile the model
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Print model structure
model.summary()


# Import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths for training and validation dataset
train_dir = r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\Horse-or-human\training'
validation_dir = r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\Horse-or-human\validation'

train_horses_dir = os.path.join(train_dir, 'horses')
train_humans_dir = os.path.join(train_dir, 'humans')
validation_horses_dir = os.path.join(validation_dir, 'horses')
validation_humans_dir = os.path.join(validation_dir, 'humans')

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print('Horse training image count : ', len(train_horses_fnames))
print('Human training image count : ', len(train_humans_fnames))
print('Horse validation image count : ', len(validation_horses_fnames))
print('Human validation image count :', len(validation_humans_fnames))

# Define ImageDataGenerator with augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True
                                   )

# NOTE THAT VALIDATION SET SHOULD NOT BE AUGMENTED
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Define train data image generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=20,
                                                              class_mode='binary',
                                                              target_size=(150, 150))

# Fit the Model
model_fit = model.fit(train_generator,
                      validation_data=validation_generator,
                      steps_per_epoch=10,
                      epochs=100,
                      validation_steps=5,
                      verbose=1,
                      callbacks=[callback])

# Plot training vs Validation Error
import matplotlib.pyplot as plt
acc = model_fit.history['accuracy']
val_acc = model_fit.history['val_accuracy']
loss = model_fit.history['loss']
val_loss = model_fit.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc=0)
plt.figure()

plt.show()