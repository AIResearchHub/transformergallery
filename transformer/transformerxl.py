

import torch
import torch.nn as nn

from .embedding import TransformerEmbedding
from .layer import XLAttentionLayer


class TransformerXL(nn.Module):
    """
    A standard Transformer module that outputs the unprocessed
    output of the last transformer layer

    Parameters:
    vocab_size (int): Vocabulary size
    max_len (int): Max length
    n_layers (int): Number of layers
    d_model (int): Dimension of transformer
    n_head (int): Number of attention heads
    p (int): Dropout probability

    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1
                 ):

        super(TransformerXL, self).__init__()
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_model = d_model

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len)

        self.layers = nn.ModuleList([XLAttentionLayer(d_model=d_model,
                                                      ffn_hidden=4 * d_model,
                                                      n_head=n_head,
                                                      p=p)
                                    for _ in range(n_layers)])

    def init_state(self, batch_size=1):
        return [torch.zeros(self.max_len, batch_size, self.d_model) for _ in range(self.n_layers)]

    def state_forward(self, state):
        """Returns next recurrent state, since standard transformer just return original state"""
        return state

    def forward(self, ids, state):
        """
        Computes transformer xl output
        Layer takes in (length, batch_size, d_model) so transpose before and after layers

        Parameters:
        ids (Tensor[batch_size, length]): tokens
        state (Tensor[batch_size, state_len, d_model]): recurrent state

        Returns:
        x (Tensor[batch_size, length, d_model]): output
        state (Tensor[batch_size, length, d_model]): next recurrent state

        """
        x = self.embedding(ids)

        x = x.transpose(0, 1)
        next_state = []
        for layer, s in zip(self.layers, state):
            print(len(s))
            x, next_s = layer(x, s)
            next_state.append(next_s)

        x = x.transpose(0, 1)

        return x, state

