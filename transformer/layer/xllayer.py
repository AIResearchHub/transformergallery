

import torch.nn as nn

from ..attention import XLAttention
from .feedforward import FeedForward


class XLAttentionLayer(nn.Module):
    """
    XL Attention Layer from Transformer XL,
    the previous hidden state is cached as mem.
    In the transformer model, the memories are stored
    in a list.
    """
    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(XLAttentionLayer, self).__init__()
        self.attention = XLAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, mem, mask=None, is_causal=False):
        """Compute the output of the transformer layer"""
        _x = x
        x = self.attention(q=x, kv=x, mem=mem, mask=mask, is_causal=is_causal)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x

