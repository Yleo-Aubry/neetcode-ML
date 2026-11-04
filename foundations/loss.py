import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        pass
        epsilon=1e-7
        n=len(y_pred)
        sum=0
        for i in range (n):
            y=y_true[i]
            p=np.clip(y_pred[i],epsilon,1-epsilon)
            sum+=y*np.log(p)+(1-y)*np.log(1-p)
        return max(0.0,round((-1/n)*sum,4))
    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        pass
        epsilon=1e-7
        n=len(y_true)
        sum=0
        for i in range(n):
            m=len(y_true[i])
            sum2=0
            for j in range(m):
                y=y_true[i][j]
                p=np.clip(y_pred[i][j],epsilon,1-epsilon)
                sum2+=y*np.log(p)
            sum+=sum2
        return max(0.0,round((-1/n)*sum,4))



