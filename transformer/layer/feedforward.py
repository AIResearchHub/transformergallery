

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    A simple feed forward network to be used in transformer layers.

    Architecture:
        Sequential(
            LayerNorm(dim)
            Linear(dim, inner_dim)
            GELU()
            Linear(inner_dim, dim)
        )

    Parameters:
    dim (int): The dimension of the input and output
    inner_dim (int): The dimension of the hidden layer
    """

    def __init__(self, dim, inner_dim):
        super(FeedForward, self).__init__()

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.ff(x)

