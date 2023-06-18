

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, AttentionLayer, MemorizingLayer


class MemorizingTransformer(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 device="cuda",
                 bsz=1,
                 **kwargs
                 ):

        super(MemorizingTransformer, self).__init__()
        self.device = device

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len,
                                              device=device)

        self.layers1 = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                     ffn_hidden=4 * d_model,
                                                     n_head=n_head,
                                                     p=p)
                                    for _ in range(n_layers)])

        self.memorizing_layer = MemorizingLayer(d_model=d_model,
                                                ffn_hidden=4 * d_model,
                                                n_head=n_head,
                                                p=p,
                                                bsz=bsz,
                                                device=device)

        self.layers2 = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                     ffn_hidden=4 * d_model,
                                                     n_head=n_head,
                                                     p=p)
                                    for _ in range(n_layers)])

        self.reset()

    def reset(self):
        self.memorizing_layer.reset()
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def forward(self, ids):
        x = self.embedding(ids)

        for layer in self.layers1:
            x = layer(x)
        x = self.memorizing_layer(x)
        for layer in self.layers1:
            x = layer(x)

        return x

