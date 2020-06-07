import numpy as np
from os import path


def hamming_distance(X, X_train):
    return (np.ones(shape=X.shape) - X) @ (X_train.T) + X.dot((np.ones(shape=X_train.shape) - X_train).T)


def sort_train_labels_knn(Dist, y):
    res = np.zeros(shape=np.shape(Dist))

    for x, r in enumerate(Dist):
        srt = np.argsort(r, kind="mergesort")
        res[x] = y[srt]
    return res


def p_y_x_knn(y, k):
    y = y[:, :k]
    res = []
    y = y.astype('int64')
    for row in y:
        res.append(np.bincount(row, minlength=10))
    res = np.array(res)
    return res / k


def classification_error(p_y_x, y_true):
    res = 0.0
    for i, r in enumerate(p_y_x):
        w1 = np.argwhere(r == np.amax(r)).flatten()
        w2 = max(w1.tolist())
        res += 1 if y_true[i] != w2 else 0

    return res / len(y_true)


def model_selection_knn(y_val, k_values, srt):
    best = [2.0, 0, []]

    for k in k_values:
        pyx = p_y_x_knn(srt, k)
        err = classification_error(pyx, y_val)
        print(err)
        if err < best[0]:
            best[0] = err
            best[1] = k
        best[2].append(err)

    return best[0], best[1], best[2]
