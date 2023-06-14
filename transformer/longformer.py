

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, LongformerLayer
from transformers import LongformerModel


class Longformer(nn.Module):
    """
    A standard Longformer module that outputs the unprocessed
    output of the last transformer layer

    Benchmarks:
    Transfomer seqlen 4096 = 6090MB
    Transformer seqlen 8192 = 19202MB

    Longformer seqlen 4096 = 5474MB
    Longformer seqlen 8192 = 9550MB

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
        super(Longformer, self).__init__()
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_model = d_model

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len)

        self.layers = nn.ModuleList(
            [LongformerLayer(d_model=d_model, ffn_hidden=4 * d_model, n_head=n_head, p=p)
             for _ in range(n_layers)])

    def init_state(self, batch_size=1, device="cpu"):
        return torch.zeros(1, batch_size, 1, 1, device=device)

    def state_forward(self, ids, state):
        """Returns next recurrent state, since standard transformer just return original state"""
        return state

    def forward(self, ids, state):
        """
        Computes transformer output

        Parameters:
        ids (Tensor[batch_size, length]): tokens
        state (Tensor[batch_size, state_len, d_model]): recurrent state

        Returns:
        x (Tensor[batch_size, length, d_model]): output
        state (Tensor[batch_size, length, d_model]): next recurrent state

        """
        x = self.embedding(ids)

        for layer in self.layers:
            x = layer(x)
            # print(x.shape)

        return x, state


class LongformerHuggingface(nn.Module):

    def __init__(self, pretrained="allenai/longformer-base-4096", **kwargs):
        super(LongformerHuggingface, self).__init__()

        self.model = LongformerModel.from_pretrained(pretrained)

    def init_state(self, batch_size=1, device="cpu"):
        return torch.zeros(1, batch_size, 1, 1, device=device)

    def state_forward(self, ids, state):
        return state

    def forward(self, ids, state):
        output = self.model(ids)

        return output.last_hidden_state, state

