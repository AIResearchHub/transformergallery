

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def sliding_chunks_matmul_qk(q, k, w):
    """Implementation of sliding chunks no overlap with one write head"""
    bsz, seqlen, nhead, dhead = q.size()
    memseqlen = k.size(1)
    assert seqlen % w == 0

    chunk_q = q.view(bsz, seqlen // w, w, nhead, dhead)
    chunk_k = k.view(bsz, memseqlen // w, w, 1, dhead)

    chunk_k_expanded = torch.stack((
        F.pad(chunk_k[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.),
        chunk_k,
        F.pad(chunk_k[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1), value=0.)
    ), dim=-1)

    assert chunk_k_expanded.shape == (bsz, memseqlen // w, w, 1, dhead, 3)

    # attn_score = torch.einsum('bhid,bojd->bhij', (q, k))

    diagonal_attn = torch.einsum('bcxhd,bzyode->bcxhey', (chunk_q, chunk_k_expanded))

    assert diagonal_attn.shape == (bsz, seqlen // w, w, nhead, 3, w)

    return diagonal_attn.reshape(bsz, seqlen, nhead, 3 * w)


def sliding_chunks_matmul_pv(prob, v, w):
    """Implementation of sliding chunks no overlap with one write head"""
    bsz, seqlen, nhead, _ = prob.shape
    _, memseqlen, _, dhead = v.shape

    chunk_prob = prob.view(bsz, seqlen // w, w, nhead, 3, w)
    chunk_v = v.view(bsz, memseqlen // w, w, 1, dhead)

    chunk_v_extended = torch.stack((
        F.pad(chunk_v[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0),
        chunk_v,
        F.pad(chunk_v[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1), value=0.0),
    ), dim=-1)

    assert chunk_v_extended.shape == (bsz, memseqlen // w, w, 1, dhead, 3)

    context = torch.einsum('bcwhpd,bzdoep->bcwhe', (chunk_prob, chunk_v_extended))

    assert context.shape == (bsz, seqlen // w, w, nhead, dhead)

    return context.reshape(bsz, seqlen, nhead, dhead)


class LocalXLAttention(nn.Module):
    """
    Attention module for Transformer layers
    """

    def __init__(self, d_model, n_head):
        super(LocalXLAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, kv, mem=None, mask=None, w=512):
        """
        Parameters:
        q:     [batch_size, length, d_model]
        kv:    [batch_size, length, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        bsz, length, d_model = q.shape

        if mem is not None:
            c = torch.concat([mem, kv], dim=1)
        else:
            c = kv

        q, k, v = self.w_q(q), *self.w_kv(c).chunk(2, dim=-1)
        q, k, v = q.view(bsz, length, self.n_head, self.d_head), k.unsqueeze(2), v.unsqueeze(2)

        q /= math.sqrt(self.d_head)

        attn_weights = sliding_chunks_matmul_qk(q, k, w=w)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -10000)
        attn_probs = F.softmax(attn_weights, dim=-1)

        out = sliding_chunks_matmul_pv(attn_probs, v, w=w)

        out = out.view(bsz, length, d_model)
        out = self.w_concat(out)

        return out

