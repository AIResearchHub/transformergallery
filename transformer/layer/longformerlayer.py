

import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerSelfAttention

from ..attention import LocalAttention
from .feedforward import FeedForward


class LongformerLayer(nn.Module):
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
        super(LongformerLayer, self).__init__()
        self.attention = LocalAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, src_mask=None):
        """Compute the output of the transformer layer"""
        _x = x
        x = self.attention(q=x, kv=x, mask=src_mask)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x


class LongformerLayerPrewritten(nn.Module):
    """
    Class representing a standard transformer layer. This layer includes Longformer self-attention,
    normalization, dropout, and a feed-forward network.

    Parameters:
    d_model (int): The dimension of the model.
    ffn_hidden (int): The size of the hidden layer in the feed-forward network.
    n_head (int): The number of attention heads.
    p (float): The probability of dropout.
    """

    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(LongformerLayerPrewritten, self).__init__()
        self.config = LongformerConfig(hidden_size=d_model,
                                       intermediate_size=ffn_hidden,
                                       num_attention_heads=n_head,
                                       hidden_dropout_prob=p,
                                       attention_probs_dropout_prob=p,
                                       attention_window=[128] * 4,
                                       )
        self.attention = LongformerSelfAttention(layer_id=0, config=self.config)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, src_mask=None):
        """
        Forward pass of the LongformerAttentionLayer.

        Parameters:
        x (Tensor): Input tensor.
        src_mask (Tensor, optional): Source mask tensor.

        Returns:
        Tensor: Output tensor after passing through the layer.
        """
        if src_mask is None:
            attention_mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.long, device=x.device)
        else:
            attention_mask = src_mask

        # In this case, it's initialized to all False, meaning no position is masked.
        is_index_masked = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)

        _x = x
        x = self.attention(x, attention_mask=attention_mask, is_index_masked=is_index_masked)[0]

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x

