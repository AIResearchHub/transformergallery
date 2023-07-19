

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class XLAttention(nn.Module):
    """
    XL Attention that incorporates previous cached keys and values.
    Main difference from standard attention is this
            attn_score = torch.einsum('bhid,bojd->bhij', (q, k))
    to adjust for different length queries and key/value pair.
    """

    def __init__(self, d_model, n_head):
        super(XLAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_concat = nn.Linear(d_model, d_model, bias=True)

    def forward(self, q, kv, mem=None, mask=None, is_causal=False):
        """
        Parameters:
        q:     [batch_size, length, d_model]
        kv:    [batch_size, length, d_model]
        mems:  [batch_size, mem_length, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        # batch_size, length, d_model = q.shape

        if mem is not None:
            c = torch.concat([mem, kv], dim=1)
            mem_length = c.size(1) - q.size(1)
        else:
            c = kv

        # q  [batch_size, length, d_model]
        # kv [batch_size, length+mem_length, d_model]
        q, k, v = self.w_q(q), self.w_k(c), self.w_v(c)
        q, k, v = self.split(q), self.split(k), self.split(v)

        if mem is not None and mask is not None:
            mask = F.pad(mask, (mem_length, 0, 0, 0), value=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.w_concat(out)

        return out

        # q /= math.sqrt(self.d_head)
        #
        # # q  [batch_size, n_head, length, d_head]
        # # k  [batch_size, n_head, length+mem_length, d_head]
        # attn_score = torch.einsum('bhid,bojd->bhij', (q, k))
        #
        # # if mask is not None:
        # #     attn_score = attn_score.masked_fill(mask == 0, -torch.finfo(attn_score.dtype).max)
        #
        # # if is_causal:
        # i, j = attn_score.shape[-2:]
        # print(attn_score.shape)
        # causal_mask = torch.ones((i, j), dtype=torch.bool, device=q.device).triu(j - i + 1)
        # print(causal_mask.to(torch.int32))
        # attn_score = attn_score.masked_fill(causal_mask, -torch.finfo(attn_score.dtype).max)
        #
        # attn_prob = F.softmax(attn_score, dim=-1)
        #
        # # attn_prob [batch_size, n_head, length, length+mem_length]
        # # v         [batch_size, n_head, length+mem_length, d_head]
        # out = (attn_prob @ v).transpose(1, 2).reshape(batch_size, length, d_model)
        # out = self.w_concat(out)
        #
        # # out [batch_size, length, d_model]
        # assert out.shape == (batch_size, length, d_model)
        #
        # return out

    def split(self, tensor):
        tensor = tensor.view(tensor.size(0), tensor.size(1), self.n_head, self.d_head)
        tensor = tensor.transpose(1, 2)

        return tensor

