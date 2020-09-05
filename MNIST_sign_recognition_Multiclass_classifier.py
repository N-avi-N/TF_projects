import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from google.colab import files

# Data source
# https://www.kaggle.com/datamunge/sign-language-mnist

# function to load csv,
# separate labels and images
# reshape image to 2d format
# export image and label as output
def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter = ',')
        first_line = True
        temp_labels = []
        temp_images = []
        for row in csv_reader:
            if first_line:
                print('ignoring first line')
                first_line = False           # converting flag to False, next lines read
            else:
                temp_labels.append(row[0])
                image_data = row[1:785]
                image_data_as_array = np.array_split(image_data, 28)
                temp_images.append(image_data_as_array)
        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')

    return images, labels

training_images, training_labels = get_data("C:\\Users\\navin\\Desktop\\PyCharm Projects\\TF projects\\MNIST sign language\\sign_mnist_train.csv")
testing_images, testing_labels = get_data('C:\\Users\\navin\\Desktop\\PyCharm Projects\\TF projects\\MNIST sign language\\sign_mnist_test.csv')

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

# add an additional dimension
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

# define the training data generator
train_datagen = ImageDataGenerator(
    rescale=1./225,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# we do not apply any transformation to the testing/validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# check input, test image shapes
print(training_images.shape)
print(testing_images.shape)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_datagen.flow(training_images, training_labels, batch_size=32),
                    steps_per_epoch=len(training_images)/32,                  # number of images processed before we consider it an epoch
                    epochs=15,
                    validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32),
                    validation_steps=len(testing_images)/32)

model.evaluate(testing_images, testing_labels)

# plot the accuracy charts
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()