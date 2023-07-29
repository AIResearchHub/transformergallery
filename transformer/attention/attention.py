

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention module for Transformer layers.
    Composes of learnable parameters in
    query, key, value and concat linear modules.
    """

    def __init__(self, d_model, n_head):
        super(Attention, self).__init__()
        self.n_head = n_head

        # We set the keys, queries and value matrices of the models.  These will be our learnable parameters.
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)

        # Furthermore, we create a concatenation layer for cleaning up the output of our multi-head attention so that it can be fed to the next layer of the transformer while preserving the flow of the gradient.
        self.w_concat = nn.Linear(d_model, d_model, bias=True)

    def forward(self, q, kv, mask=None, is_causal=False):
        """
        Parameters:
        q:     [batch_size, length, d_model]
        kv:    [batch_size, length, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        # Here, we are calculating the acutal keys queries and values using the weigths we defined in the initializaiton function
        q, k, v = self.w_q(q), self.w_k(kv), self.w_v(kv)

        # We split them to perform multi-head attention
        q, k, v = self.split(q), self.split(k), self.split(v)

        # q = q.to(torch.float16)
        # k = k.to(torch.float16)
        # v = v.to(torch.float16)

        # with torch.backends.cuda.sdp_kernel(
        #         enable_flash=True
        # ):

        # Now we send our heads through the attention formula (see the attention folder for deeper exlpanation)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

        # out = out.to(torch.float32)

        # now we concatenate the output of the different matrices of the above line and we send that through the feed forward nn to get it ready for the next attention layer.
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

        if not d_model % self.n_head == 0:
            raise Exception(f"Provided number of heads - {self.n_heads} - not a factor of the inputted tensor's dimension - {d_model}; integer division inaccurate")
        
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

        # This scary formula is essentially doing the concatenation of the multiple heads
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

