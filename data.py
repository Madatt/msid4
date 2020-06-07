import tensorflow as tf
import numpy as np
import sys
from PIL import Image

sys.modules['Image'] = Image


def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
    return x_train, x_test, y_train, y_test


def get_data_extended():
    x_train, x_test, y_train, y_test = get_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    x_new, y_new = datagen.flow(x_train, y_train, batch_size=60000)[0]
    x_train = np.concatenate([x_train, x_new], axis=0)
    y_train = np.concatenate([y_train, y_new], axis=0)

    return x_train, x_test, y_train, y_test
