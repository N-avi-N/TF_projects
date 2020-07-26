import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

# Checking the dataset
print(len(os.listdir(r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\Cat')))
print(len(os.listdir(r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\Dog')))

# Delete the pre existing folder to prevent overwriting of data
try:
    shutil.rmtree(r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\cats-v-dogs')
except OSError as e:
    pass

# Create folders for split testing and train data
try:
    os.mkdir(r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\cats-v-dogs')
    os.mkdir(r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\cats-v-dogs\training')
    os.mkdir(r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\cats-v-dogs\testing')
    os.mkdir(r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\cats-v-dogs\training\cats')
    os.mkdir(r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\cats-v-dogs\training\dogs')
    os.mkdir(r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\cats-v-dogs\testing\cats')
    os.mkdir(r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\cats-v-dogs\testing/dogs')
except OSError:
    pass


# Defining function to split data from the train and test folders
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " has zero length, hence ignoring")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))

    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


CAT_SOURCE_DIR = 'C:\\Users\\navin\\Desktop\\PyCharm Projects\\TF projects\\PetImages\\Cat\\'
TRAINING_CATS_DIR = 'C:\\Users\\navin\\Desktop\\PyCharm Projects\\TF projects\\PetImages\\cats-v-dogs\\training\\cats\\'
TESTING_CATS_DIR = 'C:\\Users\\navin\\Desktop\\PyCharm Projects\\TF projects\\PetImages\\cats-v-dogs\\testing\\cats\\'
DOG_SOURCE_DIR = 'C:\\Users\\navin\\Desktop\\PyCharm Projects\\TF projects\\PetImages\\Dog\\'
TRAINING_DOGS_DIR = 'C:\\Users\\navin\\Desktop\\PyCharm Projects\\TF projects\\PetImages\\cats-v-dogs\\training\\dogs\\'
TESTING_DOGS_DIR = 'C:\\Users\\navin\\Desktop\\PyCharm Projects\\TF projects\\PetImages\\cats-v-dogs\\testing\\dogs\\'

# create test train data folders with data
split_size = 0.9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

# Check the folder contents of the newly created test train folders
print(len(os.listdir(TRAINING_CATS_DIR)))
print(len(os.listdir(TRAINING_DOGS_DIR)))
print(len(os.listdir(TESTING_CATS_DIR)))
print(len(os.listdir(TESTING_DOGS_DIR)))

# Define the Model
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                    ])

# Compile the model
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Creating train nad test data loaders
TRAINING_DIR = r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\cats-v-dogs\training'
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\cats-v-dogs\testing'
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(150, 150))
# Fit model to the data
model_obj = model.fit_generator(train_generator,
                                epochs=50,
                                verbose=1,
                                validation_data=validation_generator)

# Print train vs test loss


import matplotlib.image as mpimage
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
# -----------------------------------------------------------
acc = model_obj.history['accuracy']
val_acc = model_obj.history['val_accuracy']
loss = model_obj.history['loss']
val_loss = model_obj.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation Loss')
plt.figure()

# Test Model
import numpy as np
from keras.preprocessing import image

image_path = r'C:\Users\navin\Desktop\PyCharm Projects\TF projects\PetImages\dog_test_image.jpg'
img = image.load_img(image_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    print("image is a dog")
else:
    print("image is a cat")
