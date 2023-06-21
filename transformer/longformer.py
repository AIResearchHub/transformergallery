

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, LongformerLayer
from transformers import LongformerModel


class Longformer(nn.Module):
    """
    Longformer is an improvement over Transformer that enables
    it to attend to long sequences without the quadratic memory
    requirements by Transformer. The sliding window attention
    allows it to scale to 10k tokens with linear memory requirements.

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
                 p=0.1,
                 **kwargs
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

        self.reset()

    def from_pretrained(self):
        pass

    def reset(self):
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def forward(self, ids):
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

        return x


class LongformerHuggingface(nn.Module):

    def __init__(self, pretrained="allenai/longformer-base-4096", **kwargs):
        super(LongformerHuggingface, self).__init__()

        self.model = LongformerModel.from_pretrained(pretrained)

    def reset(self):
        pass

    def init_state(self, batch_size=1, device="cpu"):
        return torch.zeros(1, batch_size, 1, 1, device=device)

    def forward(self, ids):
        output = self.model(ids)

        return output.last_hidden_state

