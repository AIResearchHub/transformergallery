

import torch.nn as nn

from .embedding import TransformerEmbedding
from .attention import XLAttention


class TransformerXL(nn.Module):

    def __init__(self):
        super(TransformerXL, self).__init__()
        pass

    def forward(self, x, mem=None):
        pass

