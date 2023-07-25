

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


class GRUGate(nn.Module):
    """
    GRU Gating for Gated Transformer-XL (GTrXL)

    See Stabilizing Transformer for Reinforcement Learning:
    https://arxiv.org/pdf/1910.06764v1.pdf
    """

    def __init__(self, dim, bg=0.1):
        super(GRUGate, self).__init__()
        self.Wr = nn.Linear(dim, dim)
        self.Ur = nn.Linear(dim, dim)
        self.Wz = nn.Linear(dim, dim)
        self.Uz = nn.Linear(dim, dim)
        self.Wg = nn.Linear(dim, dim)
        self.Ug = nn.Linear(dim, dim)
        self.bg = bg

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)

        return g

