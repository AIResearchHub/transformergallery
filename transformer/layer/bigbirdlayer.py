

import torch.nn as nn

from .feedforward import FeedForward

from transformers import BigBirdSelfAttention, BigBirdConfig


class BigBirdLayerPrewritten(nn.Module):
    """
    Class representing a standard transformer layer. This layer includes BigBird self-attention,
    normalization, dropout, and a feed-forward network.

    Parameters:
    d_model (int): The dimension of the model.
    ffn_hidden (int): The size of the hidden layer in the feed-forward network.
    n_head (int): The number of attention heads.
    p (float): The probability of dropout.
    """

    def __init__(self, d_model, ffn_hidden, n_head, p, block_size, num_random_blocks):
        super(BigBirdLayerPrewritten, self).__init__()
        self.config = BigBirdConfig.from_pretrained('google/bigbird-roberta-base', attention_type="block_sparse",
                                                    block_size=block_size, num_random_blocks=num_random_blocks)
        self.attention = BigBirdSelfAttention(self.config)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, src_mask=None):
        """
        Forward pass of the BigBirdAttentionLayer.

        Parameters:
        x (Tensor): Input tensor.
        src_mask (Tensor, optional): Source mask tensor.

        Returns:
        Tensor: Output tensor after passing through the layer.
        """
        _x = x
        x = self.attention(input_ids=x, attention_mask=src_mask)[0]

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x