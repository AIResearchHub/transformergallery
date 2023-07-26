

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, AttentionLayer, MemorizingLayer


class MemorizingTransformer(nn.Module):
    """
    Memorizing Transformer was proposed in
    https://github.com/lucidrains/memorizing-transformers-pytorch.
    It uses a memorizing layer that stores past keys and values
    into an index that is then retrieved via a kNN search.
    The queries are matched against keys for maximum inner product,
    and the most similar key/value pair are retrieved.
    This allows for long range memory that scales up to 300k tokens.
    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 device="cuda:0",
                 bsz=1,
                 **kwargs
                 ):

        super(MemorizingTransformer, self).__init__()
        self.device = device

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len,
                                              device=device)

        self.layers1 = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                     ffn_hidden=4 * d_model,
                                                     n_head=n_head,
                                                     p=p)
                                      for _ in range(n_layers // 2)])

        self.memorizing_layer = MemorizingLayer(d_model=d_model,
                                                ffn_hidden=4 * d_model,
                                                n_head=n_head,
                                                p=p,
                                                bsz=bsz,
                                                device=device)

        self.layers2 = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                     ffn_hidden=4 * d_model,
                                                     n_head=n_head,
                                                     p=p)
                                      for _ in range(1 - n_layers // 2)])

        self.reset()

    def reset(self):
        self.memorizing_layer.reset()
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def forward(self, ids):
        x = self.embedding(ids)

        for layer in self.layers1:
            x = layer(x)
        x = self.memorizing_layer(x)
        for layer in self.layers2:
            x = layer(x)

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
                        self.layers[layer_num].ffn.ff[1].weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("intermediate.dense.bias"):
                        self.layers[layer_num].ffn.ff[1].bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.dense.weight") and not x.endswith("attention.output.dense.bias"):
                        self.layers[layer_num].ffn.ff[3].weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.dense.bias") and not x.endswith("attention.output.dense.bias"):
                        self.layers[layer_num].ffn.ff[3].bias = nn.Parameter(state_dict[x].detach())

                    # layer norms
                    if x.endswith("attention.output.LayerNorm.weight"):
                        self.layers[layer_num].norm1.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("attention.output.LayerNorm.bias"):
                        self.layers[layer_num].norm1.bias = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.LayerNorm.weight") and not x.endswith("attention.output.LayerNorm.weight"):
                        self.layers[layer_num].norm2.weight = nn.Parameter(state_dict[x].detach())
                    if x.endswith("output.LayerNorm.bias") and not x.endswith("attention.output.LayerNorm.bias"):
                        self.layers[layer_num].norm2.bias = nn.Parameter(state_dict[x].detach())
