"""
The following code was inspired by: https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51

I thought that Mr. Odom's code representation of attention and self attention was far more simplistic and easy to
follow as compared to other models, making it perfect for understanding self-attention.

I've changed some things around as well as documented the code to make it clear what we are doing step-by step.

"""

# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
# from self_attention import SelfAttentionHead
from multi_head_attention import MultiHeadAttention
import torch


# -----------------------------------------------------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # Here I just made up numbers to see if the multi-head attention mechanism woks as intended
    # The goal of the code is not to use or train multi-head attention, but just to create it
    # therefore, the determinant of success will be if the output is valid
    number_of_batches = 3
    number_of_inputs = 24
    embedding_dimension = 512
    queries_and_keys_hidden_dimension = 1024
    values_hidden_dimension = 512
    number_of_heads = 8

    # Creating random values for keys, queries, and values
    query = torch.rand([number_of_batches, number_of_inputs, embedding_dimension])
    key = torch.rand([number_of_batches, number_of_inputs, embedding_dimension])
    value = torch.rand([number_of_batches, number_of_inputs, embedding_dimension])

    # Initializing a multi-head attention instance
    multi_head = MultiHeadAttention(number_of_heads=number_of_heads,
                                    queries_keys_hidden_dimension=queries_and_keys_hidden_dimension,
                                    embedding_dimension=embedding_dimension,
                                    values_hidden_dimension=values_hidden_dimension)

    # Seeing the result of a forward pass
    print(multi_head.forward(query, key, value).size())
