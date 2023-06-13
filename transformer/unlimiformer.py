

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, UnlimiLayer


class Unlimiformer(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 device="cuda",
                 **kwargs
                 ):

        super(Unlimiformer, self).__init__()
        self.device = device

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len,
                                              device=device)

        self.layers = nn.ModuleList([UnlimiLayer(d_model=d_model,
                                                 ffn_hidden=4 * d_model,
                                                 n_head=n_head,
                                                 p=p)
                                    for _ in range(n_layers)])

    def init_state(self):
        return torch.zeros(1, 1, 1, 1, device=self.device)

    def state_forward(self, ids, state):
        return state

    def forward(self, ids, state):
        x = self.embedding(ids)

        for layer in self.layers:
            x = layer(x)

        return x, state

