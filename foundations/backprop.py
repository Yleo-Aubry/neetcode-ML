import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        pass
        z=np.dot(x,w)+b
        pred=1/(1+np.exp(-z))
        loss = 0.5*(pred-y_true)**2
        dL_b=round((pred-y_true)*pred*(1-pred),5)
        for i in range(len(x)):
            w[i]=round(dL_b*x[i],5)
        return(w,dL_b)
