

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

    def get_init_state(self, batch_size):
        return self.init_state.unsqueeze(0).repeat(batch_size, 1, 1)

    def state_forward(self, ids, state):
        x = self.embedding(ids)

        for layer in self.layers1:
            x = layer(x)
        _, state = self.recurrent(x, state)

        return state

    def forward(self, ids, xlstate=None):
        x = self.embedding(ids)

        bsz, seqlen, dim = x.shape
        out = []
        xs = x.split(self.w, dim=-2)

        if xlstate is None:
            xlmems = []
            state = self.get_init_state(batch_size=bsz)
        else:
            xlmems, state = xlstate

        for x in xs:
            xlmems = iter(xlmems)
            nextxlmems = []

            for layer in self.layers1:
                nextxlmems.append(x.detach())
                x = layer(x, next(xlmems, None))

            x, state = self.recurrent(x, state)

            for layer in self.layers2:
                nextxlmems.append(x.detach())
                x = layer(x, next(xlmems, None))

            out.append(x)
            xlmems = nextxlmems

        out = torch.concat(out, dim=1)
        assert out.shape == (bsz, seqlen, dim)

        return out, (xlmems, state.detach())


# class BlockRecurrentTransformerPrewritten(nn.Module):
#     """
#     A standard Transformer module that outputs the unprocessed
#     output of the last transformer layer
#
#     Parameters:
#     vocab_size (int): Vocabulary size
#     max_len (int): Max length
#     n_layers (int): Number of layers
#     d_model (int): Dimension of transformer
#     n_head (int): Number of attention heads
#     p (int): Dropout probability
#
#     """
#
#     def __init__(self,
#                  vocab_size,
#                  max_len=512,
#                  n_layers=4,
#                  d_model=512,
#                  n_head=8,
#                  p=0.1,
#                  device="cuda"
#                  ):
#
#         super(BlockRecurrentTransformerPrewritten, self).__init__()
#         self.max_len = max_len
#         self.device = device
#
#         self.model = BlockRecurrentTransformerLucidrains(
#             num_tokens=vocab_size,
#             dim=d_model,
#             depth=n_layers,
#             heads=n_head,
#             max_seq_len=max_len
#         )
#
#     def init_state(self):
#         return torch.randint(0, 2000, (1, self.max_len))
#
#     def state_forward(self, state):
#         return state
#
#     def forward(self, ids, state):
#         return ids, state
#
