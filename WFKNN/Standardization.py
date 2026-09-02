import numpy as np

class MinMaxStandardizer:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mn = None
        self.mx = None

    def fit(self, X: np.ndarray):
        self.mn = X.min(axis=0)
        self.mx = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mn is None or self.mx is None:
            raise ValueError("Standardizer chưa được fit dữ liệu!")
        return (X - self.mn) / (self.mx - self.mn + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)