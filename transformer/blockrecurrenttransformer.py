

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, AttentionLayer, XLAttentionLayer, RecurrentLayer

# from block_recurrent_transformer_pytorch import BlockRecurrentTransformer as BlockRecurrentTransformerLucidrains


class BlockRecurrentTransformer(nn.Module):
    """
    Block Recurrent Transformer was proposed by this paper:
    https://arxiv.org/pdf/2203.07852.pdf
    It is composed of a recurrent attention layer
    sandwiched in between transformer xl layers.
    The recurrent layer is trained via BPTT, and it
    uses shared keys and values from inputs and recurrent states
    to share information across past inputs and current inputs

    Parameters:
        w (int): window size to iterate over per sequence
        statelen (int): Length of the recurrent state e.g. (bsz, statelen, dim)
    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 device="cuda",
                 w=512,
                 statelen=32,
                 **kwargs
                 ):
        super(BlockRecurrentTransformer, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.w = w
        self.device = device

        # learnable init state
        self.init_state = nn.Parameter(torch.randn(statelen, d_model))

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len,
                                              device=device
                                              )
        self.layers1 = nn.ModuleList([XLAttentionLayer(d_model=d_model,
                                                       ffn_hidden=4 * d_model,
                                                       n_head=n_head,
                                                       p=p
                                                       )
                                      for _ in range(n_layers // 2)])
        self.recurrent = RecurrentLayer(d_model=d_model,
                                        ffn_hidden=4 * d_model,
                                        n_head=n_head,
                                        p=p
                                        )
        self.layers2 = nn.ModuleList([XLAttentionLayer(d_model=d_model,
                                                       ffn_hidden=4 * d_model,
                                                       n_head=n_head,
                                                       p=p
                                                       )
                                      for _ in range(n_layers - (n_layers // 2))])

        self.reset()

    def reset(self):
        self.xlmems = []
        self.state = None

    def set_state(self, state=None, xlmems=None):
        if state is not None:
            self.state = state
        if xlmems is not None:
            self.xlmems = xlmems

    def get_state(self):
        return self.state, self.xlmems

    # @torch.autocast("cuda", dtype=torch.float16)
    def forward(self, ids, is_causal):
        x = self.embedding(ids)

        bsz, seqlen, dim = x.shape

        if self.state is None:
            self.state = self.init_state.unsqueeze(0).repeat(bsz, 1, 1)

        out = []
        xs = x.split(self.w, dim=-2)

        for x in xs:
            self.xlmems = iter(self.xlmems)
            nextxlmems = []

            for layer in self.layers1:
                nextxlmems.append(x.detach())
                x = layer(x, next(self.xlmems, None), is_causal=is_causal)

            x, self.state = self.recurrent(x, self.state, is_causal=is_causal)
            # x = self.recurrent(x, is_causal=is_causal)

            for layer in self.layers2:
                nextxlmems.append(x.detach())
                x = layer(x, next(self.xlmems, None), is_causal=is_causal)

            out.append(x)
            self.xlmems = nextxlmems

        out = torch.concat(out, dim=1)
        assert out.shape == (bsz, seqlen, dim)

        self.state = self.state.detach()
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

                if layer_num < self.n_layers // 2:
                    # attention
                    if x.endswith("attention.self.query.weight"):
                        self.layers1[layer_num].attention.w_q.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.query.bias"):
                        self.layers1[layer_num].attention.w_q.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.key.weight"):
                        self.layers1[layer_num].attention.w_k.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.key.bias"):
                        self.layers1[layer_num].attention.w_k.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.value.weight"):
                        self.layers1[layer_num].attention.w_v.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.value.bias"):
                        self.layers1[layer_num].attention.w_v.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.output.dense.weight"):
                        self.layers1[layer_num].attention.w_concat.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.output.dense.bias"):
                        self.layers1[layer_num].attention.w_concat.bias = nn.Parameter(state_dict[x].detach())

                    # feed forward
                    if x.endswith("intermediate.dense.weight"):
                        self.layers1[layer_num].ffn.ff[0].weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("intermediate.dense.bias"):
                        self.layers1[layer_num].ffn.ff[0].bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.dense.weight") and not x.endswith("attention.output.dense.bias"):
                        self.layers1[layer_num].ffn.ff[2].weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.dense.bias") and not x.endswith("attention.output.dense.bias"):
                        self.layers1[layer_num].ffn.ff[2].bias = nn.Parameter(state_dict[x].detach())

                    # layer norms
                    if x.endswith("attention.output.LayerNorm.weight"):
                        self.layers1[layer_num].norm1.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.output.LayerNorm.bias"):
                        self.layers1[layer_num].norm1.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.LayerNorm.weight") and not x.endswith("attention.output.LayerNorm.weight"):
                        self.layers1[layer_num].norm2.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.LayerNorm.bias") and not x.endswith("attention.output.LayerNorm.bias"):
                        self.layers1[layer_num].norm2.bias = nn.Parameter(state_dict[x].detach())

                elif layer_num < self.n_layers:
                    # attention
                    if x.endswith("attention.self.query.weight"):
                        self.layers2[layer_num - self.n_layers // 2].attention.w_q.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.query.bias"):
                        self.layers2[layer_num - self.n_layers // 2].attention.w_q.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.key.weight"):
                        self.layers2[layer_num - self.n_layers // 2].attention.w_k.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.key.bias"):
                        self.layers2[layer_num - self.n_layers // 2].attention.w_k.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.value.weight"):
                        self.layers2[layer_num - self.n_layers // 2].attention.w_v.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.self.value.bias"):
                        self.layers2[layer_num - self.n_layers // 2].attention.w_v.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.output.dense.weight"):
                        self.layers2[layer_num - self.n_layers // 2].attention.w_concat.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.output.dense.bias"):
                        self.layers2[layer_num - self.n_layers // 2].attention.w_concat.bias = nn.Parameter(state_dict[x].detach())

                    # feed forward
                    if x.endswith("intermediate.dense.weight"):
                        self.layers2[layer_num - self.n_layers // 2].ffn.ff[0].weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("intermediate.dense.bias"):
                        self.layers2[layer_num - self.n_layers // 2].ffn.ff[0].bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.dense.weight") and not x.endswith("attention.output.dense.bias"):
                        self.layers2[layer_num - self.n_layers // 2].ffn.ff[2].weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.dense.bias") and not x.endswith("attention.output.dense.bias"):
                        self.layers2[layer_num - self.n_layers // 2].ffn.ff[2].bias = nn.Parameter(state_dict[x].detach())

                    # layer norms
                    if x.endswith("attention.output.LayerNorm.weight"):
                        self.layers2[layer_num - self.n_layers // 2].norm1.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.output.LayerNorm.bias"):
                        self.layers2[layer_num - self.n_layers // 2].norm1.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.LayerNorm.weight") and not x.endswith("attention.output.LayerNorm.weight"):
                        self.layers2[layer_num - self.n_layers // 2].norm2.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.LayerNorm.bias") and not x.endswith("attention.output.LayerNorm.bias"):
                        self.layers2[layer_num - self.n_layers // 2].norm2.bias = nn.Parameter(state_dict[x].detach())

