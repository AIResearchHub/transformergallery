

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, AttentionLayer, XLAttentionLayer, RecurrentLayer

# from block_recurrent_transformer_pytorch import BlockRecurrentTransformer as BlockRecurrentTransformerLucidrains


class BlockRecurrentTransformer(nn.Module):
    """
    Block Recurrent Transformer with a recurrent attention layer
    sandwiched in between transformer xl layers
    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 w=512,
                 device="cuda",
                 xl=True,
                 statelen=32,
                 ):
        super(BlockRecurrentTransformer, self).__init__()
        self.d_model = d_model
        self.w = w
        self.device = device

        layer_cls = XLAttentionLayer if xl else AttentionLayer

        # learnable init state
        self.init_state = nn.Parameter(torch.randn(statelen, d_model))

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len,
                                              device=device
                                              )
        self.layers1 = nn.ModuleList([layer_cls(d_model=d_model,
                                                ffn_hidden=4 * d_model,
                                                n_head=n_head,
                                                p=p
                                                )
                                      for _ in range(n_layers // 2)])
        self.recurrent = RecurrentLayer(d_model=d_model,
                                        ffn_hidden=4 * d_model,
                                        n_head=n_head,
                                        p=p
                                        )
        self.layers2 = nn.ModuleList([layer_cls(d_model=d_model,
                                                ffn_hidden=4 * d_model,
                                                n_head=n_head,
                                                p=p
                                                )
                                      for _ in range(n_layers - n_layers // 2)])

        self.reset()

    def reset(self):
        self.xlmems = []
        self.state = None

    def set_state(self, state=None, xlmems=None):
        if state is not None:
            self.state = state
        if xlmems is not None:
            self.xlmems = xlmems

    def get_state(self):
        return self.state, self.xlmems

    def forward(self, ids):
        x = self.embedding(ids)

        bsz, seqlen, dim = x.shape

        if self.state is None:
            self.state = self.init_state.unsqueeze(0).repeat(bsz, 1, 1)

        out = []
        xs = x.split(self.w, dim=-2)

        for x in xs:
            self.xlmems = iter(self.xlmems)
            nextxlmems = []

            for layer in self.layers1:
                nextxlmems.append(x.detach())
                x = layer(x, next(self.xlmems, None))

            x, self.state = self.recurrent(x, self.state)

            for layer in self.layers2:
                nextxlmems.append(x.detach())
                x = layer(x, next(self.xlmems, None))

            out.append(x)
            self.xlmems = nextxlmems

        out = torch.concat(out, dim=1)
        assert out.shape == (bsz, seqlen, dim)

        return out

