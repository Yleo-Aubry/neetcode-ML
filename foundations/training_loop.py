import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        w,b=np.zeros(X.shape[1]),0.0
        for _ in range(epochs):
            pred=X @ w+b
            dw=2/X.shape[0] * (X.T @ (pred-y))
            db=2/X.shape[0] * np.sum(pred-y)
            w=w-lr*dw
            b=b-lr*db
        return (np.round(w,5),np.round(b,5))

