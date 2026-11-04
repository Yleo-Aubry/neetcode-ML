import torch
import torch.nn as nn
import math
from typing import List



class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        pass
        std=(2/(fan_in+fan_out))**(0.5)
        torch.manual_seed(0)
        return (torch.round(torch.randn(fan_out, fan_in) * std * 10000) / 10000).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        pass
        std=(2/fan_in)**0.5
        torch.manual_seed(0)
        return (torch.round(torch.randn(fan_out, fan_in) * std * 10000) / 10000).tolist()
        

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Use torch.manual_seed(0) once at the start.
        # Sample all weights in layer order with torch.randn(out_dim, in_dim) * std:
        # first (hidden_dim, input_dim), then (hidden_dim, hidden_dim).
        # Keep weights unrounded; do not call the init helpers, which reset the seed.
        # Then sample x = torch.randn(1, input_dim) without resetting the seed.
        # Apply x = torch.relu(x @ W.T) at every layer, with no biases.
        # Return the std of activations after each layer, rounded to 2 decimals.
        # Use round(x.std().item(), 2) with PyTorch's default sample std.
        pass
        L=[]
        weights=[]
        torch.manual_seed(0)
        fan_in=input_dim
        fan_out=hidden_dim
        for i in range(num_layers):

            if init_type == "xavier":
                std = (2 / (fan_in + fan_out)) ** 0.5
            elif init_type == "kaiming":
                std = (2 / fan_in) ** 0.5
            else:
                std = 1.0

            w = torch.randn(fan_out, fan_in) * std
            weights.append(w)

            fan_in = hidden_dim


        x = torch.randn(1, input_dim)
        for w in weights :
            x = torch.relu(x @ w.T)
            L.append(round(x.std().item(), 2))
        return L



