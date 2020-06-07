import numpy as np
import tensorflow.keras as keras
import data
import sys
import matplotlib.pyplot as plt
from PIL import Image
import random

sys.modules['Image'] = Image

labels = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
epochs = 10
val_split = 0.2

x_train, x_test, y_train, y_test = data.get_data_extended()

model = keras.models.load_model('best_model')
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print("Acc: " + str(test_acc))

cols = 6
rows = 6
start = random.randint(0, 5000)
x = x_test[start:start + cols * rows]
y = y_test[start:start + cols * rows]
p = model.predict(x)
pr = np.argmax(p, axis=1)
x = x.reshape((cols * rows, 28, 28))
f, arr = plt.subplots(cols, rows)
for i in range(rows):
    for j in range(cols):
        arr[i, j].imshow(x[j + i * cols])
        color = 'red'
        if pr[j + i * cols] == y[j + i * cols]:
            color = 'green'

        prob = "{:2.2f}".format(p[j + i * cols, pr[j + i * cols]] * 100.0)

        arr[i, j].set_xlabel(
            str(labels[pr[j + i * cols]]) + ": " + prob + "\n(" + str(
                labels[y[j + i * cols]]) + ")", color=color, fontsize=7)
        arr[i, j].set_xticks([])
        arr[i, j].set_yticks([])
        arr[i, j].set_xmargin(10)
        arr[i, j].set_ymargin(10)
plt.tight_layout()
plt.show()
