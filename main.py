# Imports
from keras import backend as K
from keras.layers import Dense, Conv2D, SpatialDropout2D, AveragePooling2D, BatchNormalization, LeakyReLU, Flatten, Dropout, MaxPooling2D
from keras.models import Model, Sequential
from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback
from keras.datasets import fashion_mnist
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt


# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255.
x_test = x_test / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


# Add model here


# Confusion MATRIX
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

# Display an image
plt.gray()
plt.imshow(x_train[1])