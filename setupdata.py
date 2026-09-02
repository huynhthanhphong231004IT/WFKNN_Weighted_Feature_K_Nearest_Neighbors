import numpy as np
from ucimlrepo import fetch_ucirepo

def setup_data(
    dataset, 
    train_ratio: float = 0.7, 
    random_state: int = 42,
    return_numpy: bool = True
):
    X = dataset.data.features
    y = dataset.data.targets

    if return_numpy:
        X = X.values.astype(float)
        y = y.values.ravel()

    np.random.seed(random_state)
    n_samples = len(X)
    idx = np.random.permutation(n_samples)
    n_train = int(train_ratio * n_samples)

    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    if return_numpy:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test