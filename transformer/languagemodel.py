

import torch.nn as nn
import torch.nn.functional as F


class TransformerLM(nn.Module):
    """
    Any kind of transformer cls has parameters:
        vocab_size
        max_len
        n_layers
        d_model
        n_head
        p

    and functions:
        state_forward(x, state) -> state
        forward(x,state) -> x, state

    """

    def __init__(self,
                 cls,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 device="cuda",
                 **kwargs
                 ):

        super(TransformerLM, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_head = n_head
        self.p = p

        self.transformer = cls(
            vocab_size=vocab_size,
            max_len=max_len,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            p=p,
            device=device,
            **kwargs
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

    def reset(self):
        return self.transformer.reset()

    def set_state(self, state):
        return self.transformer.set_state(state)

    def get_state(self):
        return self.transformer.get_state()

    def forward(self, x):
        x = self.transformer(x)
        x = F.log_softmax(self.lm_head(x), dim=-1)

        return x

