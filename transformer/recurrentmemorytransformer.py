

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .layer import TransformerEmbedding, AttentionLayer


class RecurrentMemoryTransformer(nn.Module):
    """
    Recurrent Memory Transformer was proposed in
    https://arxiv.org/abs/2207.06881
    The idea is that memory is represented as tokens
    that are generated at the right of output tokens.
    The mem tokens are then concatenated left and right
    of input tokens during the next step.
    The model is trained via BPTT inside sequence in
    blocks of num_token + 2*mem_tokens

    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=768,
                 n_head=8,
                 p=0.1,
                 device="cuda",
                 num_tokens=64,
                 mem_tokens=32,
                 **kwargs
                 ):
        super(RecurrentMemoryTransformer, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.device = device

        # recurrent memory transformer params
        assert max_len % num_tokens == 0

        self.num_tokens = num_tokens
        self.mem_tokens = mem_tokens

        # real context length (including memory)
        self.w = 2 * mem_tokens + num_tokens

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len,
                                              device=device)

        self.layers = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                    ffn_hidden=4 * d_model,
                                                    n_head=n_head,
                                                    p=p)
                                    for _ in range(n_layers)])

        self.reset()

        # learnable init mem tokens
        self.init_mem = nn.Parameter(torch.randn(mem_tokens, d_model))

        # custom mask
        self.custom_mask = self.create_custom_mask()

    def create_custom_mask(self):
        causal_mask = torch.ones((self.num_tokens, self.num_tokens), device=self.device, dtype=torch.bool).tril()

        causal_mask = F.pad(causal_mask, (0, self.mem_tokens, self.mem_tokens, 0), value=0)
        causal_mask = F.pad(causal_mask, (self.mem_tokens, 0, 0, self.mem_tokens), value=1)

        mask = rearrange(causal_mask, 'i j -> 1 1 i j')

        assert mask.shape == (1, 1, self.w, self.w)
        return mask

    def reset(self):
        self.mem = None

    def set_state(self, mem):
        self.mem = mem

    def get_state(self):
        return self.mem

    def forward(self, ids, is_causal=False):
        """
        Computes recurrent memory transformer output
        """
        if is_causal:
            mask = self.custom_mask

        x = self.embedding(ids)
        xs = x.split(self.num_tokens, dim=-2)
        if self.mem is None:
            self.mem = self.init_mem.unsqueeze(0)

        out = []
        for x in xs:
            x = torch.concat([self.mem, x, self.mem], axis=-2)

            for layer in self.layers:
                # modify causal mask to not include mem tokens
                x = layer(x, mask=mask, is_causal=False)

            self.mem = x[:, -self.mem_tokens:, :]
            out.append(x[:, self.mem_tokens:-self.mem_tokens, :])

        out = torch.concat(out, dim=1)
        self.mem = self.mem.detach()

        return out

    def load_pretrained(self):
        """
        load pretrained weights from huggingface transformers
        """
        assert self.d_model == 768, "dim has to be 768"
        assert (self.n_layers <= 12), "num layers exceed 12"

        from transformers import AutoModel
        pretrained = AutoModel.from_pretrained("bert-base-uncased")
        state_dict = pretrained.state_dict()

        for x in state_dict.keys():

            if x.startswith("embeddings"):
                if x.endswith("word_embeddings.weight"):
                    self.embedding.tok_emb.emb.weights = nn.Parameter(state_dict[x].detach())

            if x.startswith("encoder.layer"):
                layer_num = int(x[14])

                if layer_num < self.n_layers:
                    # attention
                    if x.endswith("attention.self.query.weight"):
                        self.layers[layer_num].attention.w_q.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.query.bias"):
                        self.layers[layer_num].attention.w_q.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.key.weight"):
                        self.layers[layer_num].attention.w_k.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.key.bias"):
                        self.layers[layer_num].attention.w_k.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.value.weight"):
                        self.layers[layer_num].attention.w_v.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.value.bias"):
                        self.layers[layer_num].attention.w_v.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.output.dense.weight"):
                        self.layers[layer_num].attention.w_concat.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.output.dense.bias"):
                        self.layers[layer_num].attention.w_concat.bias = nn.Parameter(state_dict[x].detach())

                    # feed forward
                    if x.endswith("intermediate.dense.weight"):
                        self.layers[layer_num].ffn.ff[0].weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("intermediate.dense.bias"):
                        self.layers[layer_num].ffn.ff[0].bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.dense.weight") and not x.endswith("attention.output.dense.bias"):
                        self.layers[layer_num].ffn.ff[2].weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.dense.bias") and not x.endswith("attention.output.dense.bias"):
                        self.layers[layer_num].ffn.ff[2].bias = nn.Parameter(state_dict[x].detach())

                    # layer norms
                    if x.endswith("attention.output.LayerNorm.weight"):
                        self.layers[layer_num].norm1.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.output.LayerNorm.bias"):
                        self.layers[layer_num].norm1.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.LayerNorm.weight") and not x.endswith("attention.output.LayerNorm.weight"):
                        self.layers[layer_num].norm2.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.LayerNorm.bias") and not x.endswith("attention.output.LayerNorm.bias"):
                        self.layers[layer_num].norm2.bias = nn.Parameter(state_dict[x].detach())
