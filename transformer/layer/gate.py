

import torch
import torch.nn as nn


class FixedGate(nn.Module):
    """
    Fixed Gate for block-recurrent transformer, according to paper it is the best performing gate
    Just a simple ema
    See https://arxiv.org/pdf/2203.07852.pdf (page 5) for more explanation
    """
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.bias = nn.Parameter(torch.randn(dim), requires_grad=True)

    def forward(self, x, state):
        """Computes the output of the fixed gate"""
        z = self.proj(x)
        g = torch.sigmoid(self.bias)
        return torch.mul(state, g) + torch.mul(z, 1-g)

