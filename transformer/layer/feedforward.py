

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    A simple feed forward network to be used in transformer layers.  Used to prep the multi-head attention output for the next input.

    Architecture:
        Sequential(
            LayerNorm(dim)
            Linear(dim, inner_dim)
            GELU()
            Linear(inner_dim, dim)
        )

    Args:
        dim (int): The dimension of the input and output
        inner_dim (int): The dimension of the hidden layer
    """

    def __init__(self, dim, inner_dim):
        super(FeedForward, self).__init__()

        # Simple nn with non-linear activation function in between the two layers
        # For more on GELU: https://paperswithcode.com/method/gelu#:~:text=The%20GELU%20activation%20function%20is%20x%20%CE%A6%20(%20x%20)%20%2C%20where,of%20as%20a%20smoother%20ReLU.
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    # Sending tensor x of dimension "dim" though the network
    def forward(self, x):
        return self.ff(x)

