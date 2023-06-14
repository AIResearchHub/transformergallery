

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, XLAttentionLayer
from transformers import TransfoXLModel


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
                 p=0.1,
                 device="cuda"
                 ):

        super(TransformerXL, self).__init__()
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.device = device

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len)

        self.layers = nn.ModuleList([XLAttentionLayer(d_model=d_model,
                                                      ffn_hidden=4 * d_model,
                                                      n_head=n_head,
                                                      p=p)
                                    for _ in range(n_layers)])

    def init_state(self, batch_size=1):
        return torch.zeros(self.n_layers, batch_size, self.max_len, self.d_model, device=self.device)

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

        next_state = []
        for layer, s in zip(self.layers, state):
            next_state.append(x.detach())
            x = layer(x, s)

        next_state = torch.stack(next_state)

        return x, next_state


class TransformerXLHuggingface:

    def __init__(self, pretrained="transfo-xl-wt103", **kwargs):
        super(TransformerXLHuggingface, self).__init__()

        self.model = TransfoXLModel.from_pretrained(pretrained)

    def init_state(self, batch_size=1, device="cpu"):
        return torch.zeros(1, batch_size, 1, 1, device=device)

    def state_forward(self, ids, state):
        return state

    def forward(self, ids, state):
        x = self.model(ids)

        return x, state

