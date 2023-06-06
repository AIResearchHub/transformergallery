

import torch
import torch.nn as nn

from block_recurrent_transformer_pytorch import BlockRecurrentTransformer


class Transformer(nn.Module):
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

        super(Transformer, self).__init__()
        self.max_len = max_len
        self.device = device

        self.model = BlockRecurrentTransformer(
            num_tokens=vocab_size,
            dim=d_model,
            depth=n_layers,
            heads=n_head,
            max_seq_len=max_len
        )

    def init_state(self):
        return torch.randint(0, 2000, (1, self.max_len))

    def state_forward(self, state):
        return state

    def forward(self, ids, state):

        return x, state

