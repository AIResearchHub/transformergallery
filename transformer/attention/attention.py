

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention module for Transformer layers.
    Composes of learnable parameters in
    query, key, value and concat linear modules.
    """

    def __init__(self, d_model, n_head):
        super(Attention, self).__init__()
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_concat = nn.Linear(d_model, d_model, bias=True)

        # self.w_q = nn.Linear(d_model, d_model, bias=False)
        # self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        # self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, kv, mask=None, is_causal=False):
        """
        Parameters:
        q:     [batch_size, length, d_model]
        kv:    [batch_size, length, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        q, k, v = self.w_q(q), self.w_k(kv), self.w_v(kv)
        q, k, v = self.split(q), self.split(k), self.split(v)

        # q, k, v = self.w_q(q), *self.w_kv(kv).chunk(2, dim=-1)
        # q, k, v = self.split(q), k.unsqueeze(1), v.unsqueeze(1)

        # q = q.to(torch.float16)
        # k = k.to(torch.float16)
        # v = v.to(torch.float16)

        # with torch.backends.cuda.sdp_kernel(
        #         enable_flash=True
        # ):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

        # out = out.to(torch.float32)

        out = self.concat(out)
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

    def concat(self, tensor):
        """
        Inverse function of self.split(tensor : torch.Tensor)

        Parameters:
        tensor : [batch_size, head, length, d_tensor]
        Returns:
        tensor : [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.shape
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

