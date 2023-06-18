

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class XLAttention(nn.Module):
    """
    Attention module for Transformer layers
    """

    def __init__(self, d_model, n_head):
        super(XLAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, kv, mem=None, mask=None):
        """
        Parameters:
        q:     [batch_size, length, d_model]
        kv:    [batch_size, length, d_model]
        mems:  [batch_size, mem_length, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        batch_size, length, d_model = q.shape

        if mem is not None:
            c = torch.concat([mem, kv], dim=1)
        else:
            c = kv

        # q [batch_size, length, d_model]
        # c [batch_size, length+mem_length, d_model]
        q, k, v = self.w_q(q), *self.w_kv(c).chunk(2, dim=-1)
        q, k, v = self.split(q), k.unsqueeze(1), v.unsqueeze(1)

        q /= math.sqrt(self.d_head)

        # q  [batch_size, n_head, length, d_head]
        # k  [batch_size, n_head, length+mem_length, d_head]
        attn_score = torch.einsum('bhid,bojd->bhij', (q, k))

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -10000)

        attn_prob = F.softmax(attn_score, dim=-1)

        # attn_prob [batch_size, n_head, length, length+mem_length]
        # v         [batch_size, n_head, length+mem_length, d_head]
        out = (attn_prob @ v).transpose(1, 2).reshape(batch_size, length, d_model)
        out = self.w_concat(out)

        # out [batch_size, length, d_model]
        assert out.shape == (batch_size, length, d_model)

        return out

    def split(self, tensor):
        tensor = tensor.view(tensor.size(0), tensor.size(1), self.n_head, self.d_head)
        tensor = tensor.transpose(1, 2)

        return tensor

