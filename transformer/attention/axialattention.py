

import torch
import torch.nn as nn


class AxialAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head *  heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, h, w, _, heads = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = map(lambda t: rearrange(t, 'b h w (heads d) -> b heads h w d', heads = heads), qkv.chunk(3, dim = -1))

        dots = einsum('b h i j d, b h x y d -> b h i j x y', q, k) * (1. / sqrt(k.shape[-1]))
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j x y, b h x y d -> b h i j d', attn, v)
        out = rearrange(out, 'b heads h w d -> b h w (heads d)')
        return self.to_out(out)

