import numpy as np
from sklearn.datasets import load_iris

def load_and_preprocess_iris():
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    rng = np.random.default_rng(0)
    train_idx = []
    test_idx = []

    # Стратифікований поділ 70/30: по 35 об'єктів для навчання, по 15 для тесту
    for c in [0, 1, 2]:
        c_indices = np.where(y == c)[0]
        rng.shuffle(c_indices)
        train_idx.extend(c_indices[:35])
        test_idx.extend(c_indices[35:])

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0, ddof=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test
