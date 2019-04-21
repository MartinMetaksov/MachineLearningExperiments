# %% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage

# %%
# Initialize cnn
classifier = Sequential()

# %%
# Convolution
classifier.add(Convolution2D(
    32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# %%
# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# %%
# Flattening
classifier.add(Flatten())

# %%
# Full connection
classifier.add(Dense(activation='relu', units=128))
classifier.add(Dense(activation='sigmoid', units=1))

# %%
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# %%
test_datagen = ImageDataGenerator(rescale=1./255)

# %%
training_set = train_datagen.flow_from_directory(
    'NeuralNetworks/ConvolutionalNeuralNetworks/dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# %%
test_set = test_datagen.flow_from_directory(
    'NeuralNetworks/ConvolutionalNeuralNetworks/dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# %%
classifier.fit_generator(
    training_set,
    steps_per_epoch=250,
    epochs=25,
    validation_data=test_set,
    validation_steps=2000)


# %%
