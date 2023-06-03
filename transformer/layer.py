

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import TransformerEmbedding, LearnedPositionalEncoding
from .attention import Attention, XLAttention, RecurrentAttention


class AttentionLayer(nn.Module):
    """
    Class representing a standard transformer layer. This layer includes self-attention,
    normalization, dropout, and a feed-forward network

    Parameters:
    d_model (int): The dimension of the model
    ffn_hidden (int): The size of the hidden layer in the feed forward network
    n_head (int): The number of attention heads
    p (float): The probability of dropout
    """

    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(AttentionLayer, self).__init__()
        self.attention = Attention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, src_mask=None):
        """Compute the output of the transformer layer"""
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x


class XLAttentionLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(XLAttentionLayer, self).__init__()
        self.attention = XLAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, mem, src_mask=None):
        """Compute the output of the transformer layer"""
        _x = x
        x = self.attention(q=x, kv=x, mem=mem, mask=src_mask)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x


class RecurrentAttentionLayer(nn.Module):
    """
    A recurrent transformer layer from block-recurrent transformer using RecurrentAttention
    which includes self attention and cross attention and shared keys and values

    Parameters:
    d_model (int): The dimension of the model
    ffn_hidden (int): The size of the hidden layer in the feed forward network
    n_head (int): The number of attention heads
    p (float): The probability of dropout
    """
    def __init__(self, d_model, ffn_hidden, n_head, p, max_len=512):
        super(RecurrentAttentionLayer, self).__init__()

        # learned ids
        self.state_norm = nn.LayerNorm(d_model)
        self.state_ids = LearnedPositionalEncoding(d_model=d_model, max_len=max_len)

        # attention
        self.attention = RecurrentAttention(d_model=d_model, n_head=n_head)

        # forget gates
        self.proj_gate = FixedGate(d_model)

        # linear projection
        self.x_proj = nn.Linear(2 * d_model, d_model)
        self.s_proj = nn.Linear(2 * d_model, d_model)

        # feed forward model
        self.ffn = FeedForward(d_model, inner_dim=ffn_hidden)

    def forward(self, x, s, x_mask=None, s_mask=None):
        """Compute the output of the transformer layer"""

        _x = x
        _s = s

        s = self.state_norm(s) + self.state_ids(s)

        x_proj, s_proj = self.attention(qx=x, kx=x, vx=x, qs=s, ks=s, vs=s)

        # finish computing out
        x_residual = x_proj + _x
        out = self.ffn(x_residual) + x_residual

        # fixed simple gate
        next_s = self.proj_gate(s_proj, _s)

        return out, next_s


class FeedForward(nn.Module):
    """
    A simple feed forward network to be used in transformer layers.

    Architecture:
        Sequential(
            LayerNorm(dim)
            Linear(dim, inner_dim)
            GELU()
            Linear(inner_dim, dim)
        )

    Parameters:
    dim (int): The dimension of the input and output
    inner_dim (int): The dimension of the hidden layer
    """

    def __init__(self, dim, inner_dim):
        super(FeedForward, self).__init__()

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.ff(x)


class FixedGate(nn.Module):
    """
    Fixed Gate for block-recurrent transformer, according to paper it is the best performing gate
    Just a simple ema
    See https://arxiv.org/pdf/2203.07852.pdf (page 5) for more explanation
    """
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.bias = nn.Parameter(torch.randn(dim), requires_grad=True)

    def forward(self, x, state):
        """Computes the output of the fixed gate"""
        z = self.proj(x)
        g = torch.sigmoid(self.bias)
        return torch.mul(state, g) + torch.mul(z, 1-g)

