

import torch.nn as nn
import torch.nn.functional as F


def sliding_chunks_matmul_qk():
    return 0.


def sliding_chunks_matmul_pv():
    return 0.


class SlidingWindowAttention(nn.Module):
    """
    Attention module for Transformer layers
    """

    def __init__(self, d_model, n_head):
        super(SlidingWindowAttention, self).__init__()
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, kv, mask=None):
        """
        Parameters:
        q:     [batch_size, length, d_model]
        kv:    [batch_size, length, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        q, k, v = self.w_q(q), *self.w_kv(kv).chunk(2, dim=-1)
        q, k, v = self.split(q), k.unsqueeze(1), v.unsqueeze(1)

        attn_weights = sliding_chunks_matmul_qk(q, k)
        attn_probs = F.softmax(attn_weights, dim=-1)

        out = sliding_chunks_matmul_pv(attn_probs, v)

        out = self.concat(out)
        out = self.w_concat(out)

        return out

