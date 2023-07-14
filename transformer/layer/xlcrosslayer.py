

import torch.nn as nn

from ..attention import Attention, XLAttention
from .feedforward import FeedForward


class XLCrossLayer(nn.Module):
    """
    XL Attention Layer from Transformer XL,
    the previous hidden state is cached as mem.
    In the transformer model, the memories are stored
    in a list. Modified by adding cross attention
    after self attention to take in state
    for block feedback transformer
    """
    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(XLCrossLayer, self).__init__()
        self.self_attention = XLAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.cross_attention = Attention(d_model=d_model, n_head=n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=p)

    def forward(self, x, mem, state, src_mask=None, is_causal=False):
        """Compute the output of the transformer layer"""

        # first compute xl self attention
        _x = x
        x = self.self_attention(q=x, kv=x, mem=mem, mask=src_mask, is_causal=is_causal)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # second compute cross attention with state
        _x = x
        x = self.cross_attention(q=x, kv=state, mask=src_mask, is_causal=is_causal)
        x = self.norm2(x + _x)
        x = self.dropout2(x)

        # do feedforward
        _x = x
        x = self.ffn(x)
        x = self.norm3(x + _x)
        x = self.dropout3(x)

        return x

