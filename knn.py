import data
import content
import numpy as np


x_train, x_test, y_train, y_test = data.get_data()
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

# Image thresholding
x_train = np.where(x_train > 0.5, 1, 0)
x_test = np.where(x_test > 0.5, 1, 0)
k_values = [1, 4, 9, 20, 50, 100]

# Hamming distance
h = content.hamming_distance(x_test, x_train)
# Label sorting
srt = content.sort_train_labels_knn(h, y_train)

for k in k_values:
    error_best, best_k, errors = content.model_selection_knn(y_test,
                                                             [k], srt)
    print('K: {num1} and accuracy: {num2:.4f}'.format(num1=best_k, num2=1 - error_best))
