

import torch
import torch.nn as nn

from ..attention import LocalXLAttention
from .feedforward import FeedForward


class LongformerXLLayer(nn.Module):
    """
    Class representing a standard transformer layer. This layer includes self-attention,
    normalization, dropout, and a feed-forward network

    Parameters:
    d_model (int): The dimension of the model
    ffn_hidden (int): The size of the hidden layer in the feed forward network
    n_head (int): The number of attention heads
    p (float): The probability of dropout
    """

    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(LongformerXLLayer, self).__init__()
        self.attention = LocalXLAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, mem, src_mask=None):
        """Compute the output of the transformer layer"""
        _x = x
        x = self.attention(q=x, kv=x, mem=mem, mask=src_mask)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x

