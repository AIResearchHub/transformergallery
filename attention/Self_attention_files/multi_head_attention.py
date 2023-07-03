# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
from torch import Tensor
import torch.nn as nn
from self_attention import SelfAttentionHead


# ----------------------------------------------------------------------------------------------------------------------
# MultiHeadAttention class
# ----------------------------------------------------------------------------------------------------------------------
# Lacing multiple self-attention heads together
class MultiHeadAttention(nn.Module):
    # Same params as a regular self attention head, but we have to specify how many heads we want
    def __init__(self, number_of_heads: int, embedding_dimension: int, queries_keys_hidden_dimension: int,
                 values_hidden_dimension: int):
        # Since MultiHeadAttention is a subclass of nn.module, we perform a super call to begin with
        super(MultiHeadAttention, self).__init__()

        # Creates a list of heads
        self.heads = nn.ModuleList([SelfAttentionHead(embedding_dimension, queries_keys_hidden_dimension,
                                                      values_hidden_dimension)
                                    for _ in range(number_of_heads)])

        # feed forward layer to deal with the huge concatenation matrix
        self.feed_forward_layer = nn.Linear(number_of_heads * values_hidden_dimension, embedding_dimension)

    # forward call
    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        # We basically just concatenate all the results of each head ...
        multi_head_result = torch.cat([head(query, key, value) for head in self.heads], dim=-1)

        # ... then pass it through a feed forward neural network to clean it up
        processed_multi_head_result = self.feed_forward_layer(multi_head_result)

        return processed_multi_head_result

