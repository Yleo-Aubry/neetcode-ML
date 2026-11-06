import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        output=[]
        for i in token_ids:
            output.append(embeddings[i])
        return np.round(output,5)