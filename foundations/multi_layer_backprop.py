import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        pass
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)


        L1 = [[0 for _ in range(len(W1[0]))] for _ in range(len(W1))]
        L2 = [[0 for _ in range(len(W2[0]))] for _ in range(len(W2))]
        res={"loss":0,"dW1":L1,"db1":0,"dW2":L2,"db2":0}
        z1=np.dot(W1,x)+b1
        a1=np.maximum(0.0,z1)
        z = np.dot(W2, a1) + b2
        loss=np.mean((z-y_true)**2)
        res["loss"]=round(loss,4)
        grad=2 * (z - y_true) / len(y_true)
        res["db2"]=np.round(grad,4).tolist()
        res["dW2"] = np.round(np.outer(grad, a1), 4).tolist()
        grad2=np.dot(W2.T,grad) * (z1 > 0)
        res["db1"]=np.round(grad2,4).tolist()
        res["dW1"] = np.round(np.outer(grad2, x), 4).tolist()
        return res

        
