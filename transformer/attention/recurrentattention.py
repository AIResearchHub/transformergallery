

import torch
import torch.nn as nn


class RecurrentAttention(nn.Module):
    """
    Recurrent Attention module for Block Recurrent Transformer Recurrent Layer
    See https://arxiv.org/pdf/2203.07852.pdf (page 2)
    This attention computes 4 separate queries, 2 keys and 2 values
    from input and recurrent state respectively then
    performs self attention and cross attention
    It also uses transformer xl cache memory

    Parameters:
    d_model (int): Dimension of model
    n_head (int): Number of attention heads

    """

    def __init__(self, d_model, n_head):
        super(RecurrentAttention, self).__init__()
        self.n_head = n_head

        # get q, k, v for x and state

        self.w_qx1 = nn.Linear(d_model, d_model, bias=False)
        self.w_qs1 = nn.Linear(d_model, d_model, bias=False)
        self.w_qx2 = nn.Linear(d_model, d_model, bias=False)
        self.w_qs2 = nn.Linear(d_model, d_model, bias=False)

        self.w_kvx = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_kvs = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)

        # linear projection
        self.x_proj = nn.Linear(2 * d_model, d_model)
        self.s_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, qx, kvx, qs, kvs, mask=None):
        """
        Computes recurrent attention for block recurrent transformer

        Parameters:
        qx (Tensor[batch_size, length, d_model]): input query
        kx (Tensor[batch_size, length, d_model]): input key
        vx (Tensor[batch_size, length, d_model]): input value
        qs (Tensor[batch_size, length, d_model]): state query
        ks (Tensor[batch_size, length, d_model]): state key
        vs (Tensor[batch_size, length, d_model]): state value
        """
        # compute 4 distinct queries
        qx1, qs1, qx2, qs2 = self.w_qx1(qx), self.w_qs1(qs), self.w_qx2(qx), self.w_qs2(qs)
        qx1, qs1, qx2, qs2 = self.split(qx1), self.split(qs1), self.split(qx2), self.split(qs2)

        kx, vx = self.w_kvx(kvx).chunk(2, dim=-1)
        kx, vx = kx.unsqueeze(1), vx.unsqueeze(1)

        ks, vs = self.w_kvs(kvs).chunk(2, dim=-1)
        ks, vs = ks.unsqueeze(1), vs.unsqueeze(1)

        # perform self attention and cross attention
        x, _ = self.attention(qx1, kx, vx, mask=mask)
        s, _ = self.attention(qs1, ks, vs, mask=mask)

        xs, _ = self.attention(qx2, ks, vs, mask=mask)
        sx, _ = self.attention(qs2, kx, vx, mask=mask)

        # concatenate and linear projection
        x_proj = self.concat(torch.concat((xs, x), dim=-1))
        s_proj = self.concat(torch.concat((sx, s), dim=-1))

        x_proj = self.x_proj(x_proj)
        s_proj = self.s_proj(s_proj)

        return x_proj, s_proj

    def split(self, tensor):
        """
        Split tensor into number of head

        Parameters:
        tensor : [batch_size, length, d_model]

        Returns:
        tensor : [batch_size, head, length, d_tensor]

        """
        batch_size, length, d_model = tensor.shape

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        Inverse function of self.split(tensor : torch.Tensor)

        Parameters:
        tensor : [batch_size, head, length, d_tensor]

        Returns:
        tensor : [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.shape
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

