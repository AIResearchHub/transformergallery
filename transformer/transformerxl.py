

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, XLAttentionLayer
from transformers import TransfoXLModel


class TransformerXL(nn.Module):
    """
    Transformer XL is an upgrade to Transformer that caches
    the previous keys and values to be concatenated with
    current keys and values for Extra Long sequence lengths.

    Parameters:
        vocab_size (int): Vocabulary size
        max_len (int): Max length
        n_layers (int): Number of layers
        d_model (int): Dimension of transformer
        n_head (int): Number of attention heads
        p (int): Dropout probability

    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 device="cuda",
                 **kwargs
                 ):

        super(TransformerXL, self).__init__()
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.device = device

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len,
                                              device=device)

        self.layers = nn.ModuleList([XLAttentionLayer(d_model=d_model,
                                                      ffn_hidden=4 * d_model,
                                                      n_head=n_head,
                                                      p=p)
                                    for _ in range(n_layers)])

        self.reset()

    def reset(self):
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    # @torch.autocast("cuda", dtype=torch.float16)
    def forward(self, ids, is_causal):
        """
        Computes transformer xl output
        Layer takes in (length, batch_size, d_model) so transpose before and after layers

        Parameters:
        ids (Tensor[batch_size, length]): tokens
        state (Tensor[batch_size, state_len, d_model]): recurrent state

        Returns:
        x (Tensor[batch_size, length, d_model]): output
        state (Tensor[batch_size, length, d_model]): next recurrent state

        """
        bsz = ids.size(0)

        if self.state is None:
            self.state = torch.zeros(self.n_layers, bsz, self.max_len, self.d_model, device=self.device)

        x = self.embedding(ids)

        next_state = []
        for layer, s in zip(self.layers, self.state):
            next_state.append(x.detach())
            x = layer(x, s, is_causal=is_causal)

        self.state = torch.stack(next_state)

        return x

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


class TransformerXLHuggingface:

    def __init__(self, pretrained="transfo-xl-wt103", **kwargs):
        super(TransformerXLHuggingface, self).__init__()

        self.model = TransfoXLModel.from_pretrained(pretrained)

    def init_state(self, batch_size=1, device="cpu"):
        return torch.zeros(1, batch_size, 1, 1, device=device)

    def state_forward(self, ids, state):
        return state

    def forward(self, ids, state):
        x = self.model(ids)

        return x, state

