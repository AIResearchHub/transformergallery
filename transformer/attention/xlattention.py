

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

    def forward(self, q, kv, mems=None, mask=None):
        """
        Parameters:
        q:     [length, batch_size, d_model]
        kv:    [length, batch_size, d_model]
        mems:  [mem_length, batch_size, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        if mems is not None:
            c = torch.concat([mems, kv], dim=0)
        else:
            c = kv

        # q [length, batch_size, d_model]
        # c [length + mem_length, batch_size, d_model]
        q, k, v = self.w_q(q), self.w_k(c), self.w_v(c)
        q, k, v = self.split(q), self.split(k), self.split(v)

        # q  [length, batch_size, n_head, d_head]
        # kv [length+mem_length, batch_size, n_head, d_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (q, k)) / math.sqrt(self.d_head)

        # attn_score [length, length+mem_length, batch_size, n_head]
        attn_prob = F.softmax(attn_score, dim=1)

        # attn_prob [length, length + mem_length, batch_size, n_head]
        # v         [length, batch_size, n_head, d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, v))

        # attn_vec [length, batch_size, n_head, d_head]
        out = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        out = self.w_concat(out)

        # out [length, batch_size, d_model]
        return out

    def split(self, tensor):
        return tensor.view(tensor.size(0), tensor.size(1), self.n_head, self.d_head)

