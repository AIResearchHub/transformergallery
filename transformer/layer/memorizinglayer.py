

import torch.nn as nn

from ..attention import Attention
from ..attention import KNNAttention
from .feedforward import FeedForward


class MemorizingLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, p, bsz, device):
        super(MemorizingLayer, self).__init__()
        self.attention = Attention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.knn_attention = KNNAttention(d_model=d_model,
                                          n_head=n_head,
                                          bsz=bsz,
                                          device=device)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=p)

    def reset(self):
        self.knn_attention.reset()

    def forward(self, x, src_mask=None):
        """Compute the output of the transformer layer"""
        _x = x
        x = self.knn_attention(q=x, kv=x, mask=src_mask)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        # compute feed forward
        _x = x
        x = self.ffn(x)

        x = self.norm3(x + _x)
        x = self.dropout3(x)

        return x
