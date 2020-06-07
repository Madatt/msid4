import utils.mnist_reader as mnist_reader
import data
import content
import numpy as np

X_train, X_test, y_train, y_test = data.get_data()
X_train = np.where(X_train > 0.5, 1, 0)
X_test = np.where(X_test > 0.5, 1, 0)
k_values = [1, 4, 9, 20, 50, 100]

h = content.hamming_distance(X_test, X_train)
srt = content.sort_train_labels_knn(h, y_train)

for k in k_values:
    error_best, best_k, errors = content.model_selection_knn(y_test,
                                                             [k], srt)
    print('K: {num1} i blad: {num2:.4f}'.format(num1=best_k, num2=1 - error_best))
