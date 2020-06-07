import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import data
import sys
from PIL import Image
sys.modules['Image'] = Image

parameters = [
    [128, 0.4, 0.5, 256, 128],
    [128, 0.4, 0.5, 128, 128],
    [128, 0.4, 0.5, 128, 256],
    [128, 0.5, 0.6, 128, 128],
    [256, 0.4, 0.5, 128, 128],
    [256, 0.4, 0.5, 256, 128],
    [256, 0.5, 0.6, 256, 128],
    [128, 0.4, 0.5, 512, 128],
]
epochs = 10
val_split = 0.2

results = []
x_train, x_test, y_train, y_test = data.get_data_extended()

c = 0
for p in parameters:
    c += 1
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(),

        keras.layers.TimeDistributed(keras.layers.LSTM(p[3])),
        keras.layers.Dropout(p[1]),

        keras.layers.Flatten(),
        keras.layers.Dense(p[4], activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Dropout(p[2]),
        keras.layers.Dense(10, activation=keras.activations.softmax, kernel_initializer='he_uniform'),
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    results.append([])
    model.fit(x_train, y_train, epochs=epochs, batch_size=p[0], verbose=0, validation_split=val_split)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    results[-1].append(test_acc)
    print(str(p) + ": " + str(results[-1]))