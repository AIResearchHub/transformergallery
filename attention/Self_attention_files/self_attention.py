# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor


# ----------------------------------------------------------------------------------------------------------------------
# Static methods
# ----------------------------------------------------------------------------------------------------------------------
# Scaled dot product attention is generally how we calculate self attention, although there are other methods
# PARAMS
# queries - tensor of shape (batch_size, sequence_length, number_of features_for_queries)
# keys - tensor of shape (batch_size, sequence_length, number_of features_for_keys)
# Note: number_of features_for_keys = number_of features_for_queries
# values - tensor of shape (batch_size, sequence_length, number_of features_for_values)
def scaled_dot_product_attention(queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
    # First, we batch matrix multiply the queries vector with the transpose of the keys vector
    # This step essentially determines how important each position is relative to each other
    # Results in a sequence_length x sequence_length matrix
    matrix_mult_of_queries_and_keys = queries.bmm(keys.transpose(1, 2))

    # We need to ensure that back propagation runs smoothly
    # We keep the values of our above matrix multiplication in check by dividing by the square root
    # of the number of hidden dimension of the queries (which equals the hidden dimension of the keys)
    scalar_for_gradient_stability = queries.size(-1) ** (1/2)

    # Dividing the values of the matrix multiplication by the number of hidden dimension of the queries
    matrix_multiplication_adjusted_by_scalar = matrix_mult_of_queries_and_keys / scalar_for_gradient_stability

    # To further help gradient descent and to standardize weights, we apply softmax row-wise on out result
    softmax_scaled_matrix_multiplication = f.softmax(matrix_multiplication_adjusted_by_scalar, dim=-1)

    # Lastly, we perform batch matrix multiplication with the value matrices
    scaled_dot_product_attention_result = softmax_scaled_matrix_multiplication.bmm(values)

    # Results in a sequence_length x number_of features_for_values matrix
    return scaled_dot_product_attention_result


# ----------------------------------------------------------------------------------------------------------------------
# Self attention class
# ----------------------------------------------------------------------------------------------------------------------
class SelfAttentionHead(nn.Module):
    # PARAMS
    # Weight matrices are crucial for the attention model to learn what is important and what isn't
    # All the parameters make it so that matrix multiplication of with the queries, keys, and values goes smoothly
    # Considering the queries, keys, and values matrices will be sequence_length x embedding_length
    # we use the embedding length as the number of rows and chose a number for the amount of hidden dimensions
    def __init__(self, embedding_dimension: int, queries_keys_hidden_dimension: int, values_hidden_dimension: int):
        # Since SelfAttentionHead is a subclass of nn.module, we need to make a super call
        super(SelfAttentionHead, self).__init__()

        # **BETTER EXPLANATION NEEDED**
        # "Wait I thought we were doing matrix multiplication?  Why are we defining neural networks?"
        # Neural networks work as our weight matrix here since their weights represent linear transformations
        # between nodes
        # If you think about the edges between nodes as the entries of a matrix, it starts to become clear why we can
        # use neural networks instead of straight tensors
        # Neural networks bring us more representation and flexibility over tensors
        # Note that the input layer to the linear layer is the embedding dimension and that
        # the hidden dimensions are the output
        self.query_weights = nn.Linear(embedding_dimension, queries_keys_hidden_dimension)
        self.key_weights = nn.Linear(embedding_dimension, queries_keys_hidden_dimension)
        self.value_weights = nn.Linear(embedding_dimension, values_hidden_dimension)

    # Overriding nn.Module's forward method
    # PARAMS
    # query, key, and values are all have size input_sequence_length x embedding size
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # performing matrix multiplication with the weights
        weighted_query = self.query_weights(query)
        weighted_key = self.key_weights(key)
        weighted_value = self.value_weights(value)

        # using scaled dot product attention to find the attention weights
        attention = scaled_dot_product_attention(weighted_query, weighted_key, weighted_value)

        return attention
