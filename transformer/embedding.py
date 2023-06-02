

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Compute sinusoid encoding from original transformer paper.

    Parameters:
    d_model (int): dimension of model
    max_len (int): max length of transformer
    """
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model).cuda()
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len).cuda()
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).cuda().float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        """Obtain positional encoding according to input size"""
        batch_size, seq_len, d_model = x.size()
        return self.encoding[:seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Embedding

    Parameters:
    d_model (int): Dimension of model
    max_len (int): Max length of transformer
    """

    def __init__(self, d_model, max_len):
        super(LearnedPositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model),
                                     requires_grad=True)

    def forward(self, x):
        """Return learned positional encoding according to input shape"""
        batch_size, seq_len, d_model = x.size()
        return self.encoding[:seq_len, :]


class TokenEmbedding(nn.Module):
    """
    Token Embedding for transformer
    """

    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, ids):
        """
        Parameters:
        ids : [batch_size, length]
        token_emb : [batch_size, length, dim]
        """
        token_emb = self.emb(ids)
        return token_emb


class TransformerEmbedding(nn.Module):
    """
    Transformer Embedding, combining positional encoding and token embedding
    """

    def __init__(self, vocab_size, d_model, max_len):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)

    def forward(self, x):
        """
        Returns complete transformer embedding for transformer layers

        Parameters:
        x : [batch_size, length]

        Returns:
        token_emb + pos_emb : [batch_size, length, dim]
        """
        token_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(token_emb)
        return token_emb + pos_emb

