

import torch.nn as nn

from ..attention import Attention
from ..attention import KNNAttention
from .feedforward import FeedForward


class MemorizingLayer(nn.Module):
    """
    The kNN search layer from Memorizing Trasnformer
    Currently takes has long runtime, need to optimize
    the kNN attention in the future for shorter runtime.
    """

    def __init__(self, d_model, ffn_hidden, n_head, p, bsz, device):
        super(MemorizingLayer, self).__init__()
        self.knn_attention = KNNAttention(d_model=d_model,
                                          n_head=n_head,
                                          bsz=bsz,
                                          device=device)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def reset(self):
        self.knn_attention.reset()

    def forward(self, x, src_mask=None):
        """Compute the output of the transformer layer"""
        _x = x
        x = self.knn_attention(q=x, kv=x, mask=src_mask)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # compute feed forward
        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x
