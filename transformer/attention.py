
import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot Product Attention for Transformers

    Parameters:
    Query : [batch_size, head, length, d_tensor]
    Key : T [batch_size, head, d_tensor, length]
    Value : [batch_size, head, length, d_tensor]

    score : [batch_size, head, length, length]
    v_out : [batch_size, head, length, d_tensor]
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = q.shape
        k_t = k.transpose(2, 3)

        score = torch.matmul(q, k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        score = self.softmax(score)
        v = torch.matmul(score, v)

        return v, score


class Attention(nn.Module):
    """
    Attention module for Transformer layers
    """

    def __init__(self, d_model, n_head):
        super(Attention, self).__init__()
        self.n_head = n_head
        self.attention = ScaledDotProductAttention()

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

        # self.w_q = nn.Linear(d_model, d_model, bias=False)
        # self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        # self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        """
        Parameters:
        q:     [batch_size, length, d_model]
        kv:    [batch_size, length, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        # q, k, v = self.w_q(q), *self.w_kv(kv).chunk(2, dim=-1)
        # q, k, v = self.split(q), k.unsqueeze(1), v.unsqueeze(1)

        out, attention = self.attention(q, k, v, mask=mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out

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


class XLAttention(nn.Module):
    """
    Attention module for Transformer layers
    """

    def __init__(self, d_model, n_head):
        super(XLAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.attention = ScaledDotProductAttention()

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

        # self.w_q = nn.Linear(d_model, d_model, bias=False)
        # self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        # self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, kv, mem=None, mask=None):
        """
        Parameters:
        q:     [length, batch_size, d_model]
        kv:    [length, batch_size, d_model]
        mem:  [mem_length, batch_size, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        if mem is not None:
            c = torch.concat([mem, kv], dim=0)
        else:
            c = kv

        # q [length, batch_size, d_model]
        # c [length + mem_length, batch_size, d_model]
        q, k, v = self.w_q(q), self.w_k(c), self.w_v(c)
        q, k, v = self.split(q), self.split(k), self.split(v)

        # q  [length, batch_size, n_head, d_head]
        # kv [length+mem_length, batch_size, n_head, d_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (q, k)) / math.sqrt(self.d_head)

        # attn_score [length, length+mem_length, batch_size, n_head]
        attn_prob = F.softmax(attn_score, dim=1)

        # attn_prob [length, length + mem_length, batch_size, n_head]
        # v         [length, batch_size, n_head, d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, v))

        # attn_vec [length, batch_size, n_head, d_head]
        out = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        out = self.w_concat(out)

        # out [length, batch_size, d_model]
        return out

    def split(self, tensor):
        return tensor.view(tensor.size(0), tensor.size(1), self.n_head, self.d_head)


class LongformerAttention(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


class RecurrentAttention(nn.Module):
    """
    Recurrent Attention module for Block Recurrent Transformer Recurrent Layer
    See https://arxiv.org/pdf/2203.07852.pdf (page 2)
    This attention computes 4 separate queries, 2 keys and 2 values
    from input and recurrent state respectively then
    performs self attention and cross attention

    Parameters:
    d_model (int): Dimension of model
    n_head (int): Number of attention heads

    """

    def __init__(self, d_model, n_head):
        super(RecurrentAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaledDotProductAttention()

        # get q, k, v for x and state

        self.w_qx1 = nn.Linear(d_model, d_model)
        self.w_qs1 = nn.Linear(d_model, d_model)
        self.w_qx2 = nn.Linear(d_model, d_model)
        self.w_qs2 = nn.Linear(d_model, d_model)

        self.w_kx = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vx = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)

        # self.w_qx1 = nn.Linear(d_model, d_model, bias=False)
        # self.w_qs1 = nn.Linear(d_model, d_model, bias=False)
        # self.w_qx2 = nn.Linear(d_model, d_model, bias=False)
        # self.w_qs2 = nn.Linear(d_model, d_model, bias=False)
        #
        # self.w_kvx = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        # self.w_kvs = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)

        # linear projection
        self.x_proj = nn.Linear(2 * d_model, d_model)
        self.s_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, qx, kx, vx, qs, ks, vs, mask=None):
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

        # compute shared keys and values
        kx, vx, ks, vs = self.w_kx(kx), self.w_vx(vx), self.w_ks(ks), self.w_vs(vs)
        kx, vx, ks, vs = self.split(kx), self.split(vx), self.split(ks), self.split(vs)

        # kx, vx = self.w_kvx(kvx).chunk(2, dim=-1)
        # kx, vx = kx.unsqueeze(1), vx.unsqueeze(1)
        #
        # ks, vs = self.w_kvs(kvs).chunk(2, dim=-1)
        # ks, vs = ks.unsqueeze(1), vs.unsqueeze(1)

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

