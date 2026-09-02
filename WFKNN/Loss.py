import numpy as np

class HingeLossCalculator:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def compute_loss(self, dp: float, dn: float) -> float:
        return max(0.0, 1.0 + dp - dn)

    def compute_gradient(
        self, 
        xq: np.ndarray, 
        xp: np.ndarray, 
        xn: np.ndarray, 
        dp: float, 
        dn: float, 
        sigma2: np.ndarray
    ) -> np.ndarray:
        grad = ((xq - xp)**2 / (2 * dp * sigma2 + self.eps) - 
                (xq - xn)**2 / (2 * dn * sigma2 + self.eps))
        return grad