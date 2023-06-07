

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def sliding_chunks_matmul_qk(q, k, w):
    """Implementation of sliding chunks no overlap with one write head"""
    bsz, seqlen, nhead, dhead = q.size()
    assert seqlen % w == 0

    chunk_q = q.view(bsz, seqlen // w, w, nhead, dhead)
    chunk_k = k.view(bsz, seqlen // w, w, 1, dhead)

    chunk_k_expanded = torch.stack((
        F.pad(chunk_k[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.),
        chunk_k,
        F.pad(chunk_k[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1), value=0.)
    ), dim=-1)

    assert chunk_k_expanded.shape == (bsz, seqlen // w, w, 1, dhead, 3)

    diagonal_attn = torch.einsum('bcxhd,bcyode->bcxhey', (chunk_q, chunk_k_expanded))

    assert diagonal_attn.shape == (bsz, seqlen // w, w, nhead, 3, w)

    return diagonal_attn.reshape(bsz, seqlen, nhead, 3 * w)


def sliding_chunks_matmul_pv(prob, v, w):
    """Implementation of sliding chunks no overlap with one write head"""
    bsz, seqlen, _, dhead = v.size()
    nhead = prob.size(2)

    chunk_prob = prob.view(bsz, seqlen // w, w, nhead, 3, w)
    chunk_v = v.view(bsz, seqlen // w, w, 1, dhead)

    chunk_v_extended = torch.stack((
        F.pad(chunk_v[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0),
        chunk_v,
        F.pad(chunk_v[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1), value=0.0),
    ), dim=-1)

    assert chunk_v_extended.shape == (bsz, seqlen // w, w, 1, dhead, 3)

    context = torch.einsum('bcwhpd,bcdoep->bcwhe', (chunk_prob, chunk_v_extended))

    assert context.shape == (bsz, seqlen // w, w, nhead, dhead)

    return context.reshape(bsz, seqlen, nhead, dhead)


class LocalAttention(nn.Module):
    """
    Attention module for Transformer layers
    """

    def __init__(self, d_model, n_head):
        super(LocalAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, kv, mask=None, w=512):
        """
        Parameters:
        q:     [batch_size, length, d_model]
        kv:    [batch_size, length, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        bsz, length, d_model = q.shape

        q, k, v = self.w_q(q), *self.w_kv(kv).chunk(2, dim=-1)
        q, k, v = q.view(bsz, length, self.n_head, self.d_head), k.unsqueeze(2), v.unsqueeze(2)

        q /= math.sqrt(self.d_head)

        attn_weights = sliding_chunks_matmul_qk(q, k, w=w)
        attn_probs = F.softmax(attn_weights, dim=-1)

        out = sliding_chunks_matmul_pv(attn_probs, v, w=w)

        out = out.view(bsz, length, d_model)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        Split tensor into number of head

        Parameters:
        tensor : [batch_size, length, d_model]

        Returns:
        tensor : [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.shape

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

