

import torch.nn as nn

# See the Attention and FeedForward files to see underlying code
from ..attention import Attention        # https://github.com/AIResearchHub/transformergallery/blob/main/transformer/attention/attention.py
from .feedforward import FeedForward     # https://github.com/AIResearchHub/transformergallery/blob/main/transformer/layer/feedforward.py

# The following class represents a block of a transformer
class AttentionLayer(nn.Module):
    """
    Class representing a standard transformer layer. This layer includes self-attention,
    normalization, dropout, and a feed-forward network

    Args:
        d_model (int): The dimension of the model
        ffn_hidden (int): The size of the hidden layer in the feed forward network
        n_head (int): The number of attention heads
        p (float): The probability of dropout
    """
    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(AttentionLayer, self).__init__()

        # Here, we are using attention.py to set up our attention
        self.attention = Attention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        # Next, we use feedforward.py to instantiate the feed forward layer that is found at the end of every transformer block to prepare it for the next block.
        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    """
    Compute the output of the transformer layer

    Args:
        x (Tensor of seq_length x d_model): input
        mask (Tensor that mirrors sizes of x if not None): allows for masked multi-head attention, which takes place in the decoder
        is_casual (bool): allows for casual attention masking in scaled dot product attention   
    """
    def forward(self, x, mask=None, is_causal=False):
        # Setting a copy for the later add and norm
        _x = x

        # Putting the input through multi-head attention
        x = self.attention(q=x, kv=x, mask=mask, is_causal=is_causal)

        # Add and norm for gradient smoothness, and perform dropout for performance
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # Putting the attention output through a feed forward nn to prepare the output for the next block
        
        # Again, saving a copy of the pre-function input to add and norm later
        _x = x

        # Send post-attention output through feed forward neural network
        x = self.ffn(x)

        # Add and Norm + dropout
        x = self.norm2(x + _x)
        x = self.dropout2(x)

        # end of transformer block
        return x
