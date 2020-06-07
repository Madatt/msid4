import numpy as np
import tensorflow.keras as keras
import data
import sys
import matplotlib.pyplot as plt
from PIL import Image
import random

sys.modules['Image'] = Image

# All clothing categories
labels = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
epochs = 10
val_split = 0.2

x_train, x_test, y_train, y_test = data.get_data_extended()

model = keras.models.load_model('best_model')
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print("Overall accuracy: " + str(test_acc))

oks = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
errs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
all_p = model.predict(x_test)
all_pr = np.argmax(all_p, axis=1)

for i, xx in enumerate(x_test):
    oks[y_test[i]] += all_pr[i] == y_test[i]
    errs[y_test[i]] += all_pr[i] != y_test[i]

res = np.divide(oks, np.add(oks, errs))

print("Accuracy for each category: ")
for i, l in enumerate(labels):
    print(l + ": " + str(res[i] * 100.0) + "%")


cols = 6
rows = 6
start = random.randint(0, 5000)
x = x_test[start:start + cols * rows]
y = y_test[start:start + cols * rows]
p = model.predict(x)
pr = np.argmax(p, axis=1)
x = x.reshape((cols * rows, 28, 28))
f, arr = plt.subplots(cols, rows)
# This complicated-looking loop will display random images in a grid and label them.
# Label color corresponds to if the image was classified successfully.
for i in range(rows):
    for j in range(cols):
        arr[i, j].imshow(x[j + i * cols])
        color = 'red'
        if pr[j + i * cols] == y[j + i * cols]:
            color = 'green'

        prob = "{:2.2f}".format(
            p[j + i * cols, pr[j + i * cols]] * 100.0)  # We also display probability of the selected label

        arr[i, j].set_xlabel(
            str(labels[pr[j + i * cols]]) + ": " + prob + "\n(" + str(
                labels[y[j + i * cols]]) + ")", color=color, fontsize=7)
        arr[i, j].set_xticks([])
        arr[i, j].set_yticks([])
        arr[i, j].set_xmargin(10)
        arr[i, j].set_ymargin(10)
plt.tight_layout()
plt.show()

