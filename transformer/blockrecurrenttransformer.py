

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, AttentionLayer, RecurrentLayer

from block_recurrent_transformer_pytorch import BlockRecurrentTransformer as BlockRecurrentTransformerLucidrains


class BlockRecurrentTransformer(nn.Module):
    """
    Block Recurrent Transformer with a recurrent attention layer sandwiched in between
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
        super(BlockRecurrentTransformer, self).__init__()
        self.d_model = d_model
        self.device = device

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len
                                              )
        self.layers1 = nn.ModuleList([AttentionLayer(d_model=d_model,
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
        self.layers2 = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                     ffn_hidden=4 * d_model,
                                                     n_head=n_head,
                                                     p=p
                                                     )
                                      for _ in range(n_layers - n_layers // 2)])

    def init_state(self, batch_size, state_len):
        return torch.randn(batch_size, state_len, self.d_model, device=self.device)

    def state_forward(self, ids, state):
        x = self.embedding(ids)
        for layer in self.layers1:
            x = layer(x)
        _, state = self.recurrent(x, state)

        return state

    def forward(self, ids, state):
        x = self.embedding(ids)
        for layer in self.layers1:
            x = layer(x)
        x, state = self.recurrent(x, state)
        for layer in self.layers2:
            x = layer(x)
        return x, state


class BlockRecurrentTransformerPrewritten(nn.Module):
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

        super(BlockRecurrentTransformerPrewritten, self).__init__()
        self.max_len = max_len
        self.device = device

        self.model = BlockRecurrentTransformerLucidrains(
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
        return ids, state

