import numpy as np

class DistanceCalculator:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def euclidean(self, xq: np.ndarray, xi: np.ndarray) -> float:
        return float(np.linalg.norm(xq - xi))

    def mahalanobis(self, x1: np.ndarray, x2: np.ndarray, w: np.ndarray, sigma2: np.ndarray) -> float:
        diff = x1 - x2
        val = np.sum(w * (diff ** 2) / (sigma2 + self.eps))
        return float(np.sqrt(np.maximum(0.0, val)))