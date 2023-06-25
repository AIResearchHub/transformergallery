

import torch.nn as nn

from ..attention import RecurrentAttention
from .feedforward import FeedForward


class RecurrentLayer(nn.Module):
    """
    A recurrent layer from Block Recurrent Transformer.
    Trained via BPTT. If window size = 512 and sequence
    length = 1024, then its length 2 BPTT.

    """
    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(RecurrentLayer, self).__init__()
        self.attention = RecurrentAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, state, src_mask=None, is_causal=False):
        """Compute the output of the transformer layer"""
        _x = x
        x, state = self.attention(qx=x, kvx=x, qs=state, kvs=state, mask=src_mask, is_causal=is_causal)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x, state
