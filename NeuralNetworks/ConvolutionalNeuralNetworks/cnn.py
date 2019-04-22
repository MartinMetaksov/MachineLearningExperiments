# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage
from PIL import Image
import numpy as np
from skimage import transform

# %%
# Initialize cnn
classifier = Sequential()

# %%
# Convolution
classifier.add(Convolution2D(
    64, 3, 3, input_shape=(128, 128, 3), activation='relu'))

# %%
# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# %%
# Convolution
classifier.add(Convolution2D(128, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# %%
# Flattening
classifier.add(Flatten())
classifier.add(Dropout(0.5))

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
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary')

# %%
test_set = test_datagen.flow_from_directory(
    'NeuralNetworks/ConvolutionalNeuralNetworks/dataset/test_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary')

# %%
classifier.fit_generator(
    training_set,
    steps_per_epoch=250,
    epochs=25,
    validation_data=test_set,
    validation_steps=63)


# %%
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (128, 128, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

#%%
