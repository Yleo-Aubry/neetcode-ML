import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        pass
        n=max(z)
        norm=0
        for x in z :
            norm+=np.exp(x-n)
        return [round(np.exp(x-n)/norm,4) for x in z]
