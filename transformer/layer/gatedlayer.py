

import torch.nn as nn

from ..attention import Attention
from .feedforward import FeedForward
from .gate import GRUGate


class GatedLayer(nn.Module):
    """
    Class representing a transformer layer from Stabilizing Transformer
    for Reinforcement Learning: https://arxiv.org/pdf/1910.06764.pdf
    The idea is to perform layer norm before attention and ffn to
    enable an identity map? and quote "the Identity Map Reordering aids
    policy optimization because it initializes the agent close to a
    Markovian policy / value function"

    Parameters:
        d_model (int): The dimension of the model
        ffn_hidden (int): The size of the hidden layer in the feed forward network
        n_head (int): The number of attention heads
        p (float): The probability of dropout
    """

    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(GatedLayer, self).__init__()
        self.attention = Attention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

        # GRU Gates
        self.gate1 = GRUGate(d_model)
        self.gate2 = GRUGate(d_model)

    def forward(self, x, src_mask=None, is_causal=False):
        """Compute the output of the transformer layer"""
        _x = x
        x = self.norm1(x)
        x = self.attention(q=x, kv=x, mask=src_mask, is_causal=is_causal)

        x = self.gate1(_x, x)
        x = self.dropout1(x)

        _x = x
        x = self.norm2(x)
        x = self.ffn(x)

        x = self.gate2(_x, x)
        x = self.dropout2(x)

        return x
