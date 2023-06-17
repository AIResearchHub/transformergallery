

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import faiss
import numpy as np
from joblib import Parallel, delayed, cpu_count


class KNNAttention(nn.Module):
    """
    An attention module that contains a kNN memory
    (idea from memorizing transformer and unlimiformer)
    Instead of standard attention with computed q, k, v
    the q is searched against past cached hidden states
    to find the most similar hidden states that is then
    used to compute k and v
    """

    def __init__(self, d_model, n_head, bsz, device):
        super(KNNAttention, self).__init__()
        self.n_head = n_head
        self.bsz = bsz

        self.memory = KNNMemory(d_model, bsz, device)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def reset(self):
        self.memory.reset()

    def forward(self, q, kv, mask=None, topk=1):
        """
        get queries to compute, and keys and values hidden states to store
        in the knn memory

        Parameters:
            q (bsz, seqlen, dim): hidden states to compute queries
            kv (bsz, seqlen, dim): hidden states to store in knn memory
        """
        # q, k, v = self.w_q(q), *self.w_kv(kv).chunk(2, dim=-1)
        # q, k, v = rearrange(q, "b l (h d) -> b h l d"), k.unsqueeze(1), v.unsqueeze(1)

        q = self.w_q(q)

        # knn memory
        # same keys and values for each attention head
        # so only need to search once per token
        self.memory.add(kv)
        mem = self.memory.search(q, topk=topk)

        # is this efficient?
        mem_k, mem_v = self.w_kv(mem).chunk(2, dim=-1)

        q, k, v = rearrange(q, "b l (h d) -> b h l d", h=self.n_head), mem_k.unsqueeze(1), mem_v.unsqueeze(1)

        # attention reformulation here?
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.w_concat(out)

        return out


class KNNMemory:
    """
    TODO: implement Parallel search
    """

    def __init__(self, dim, bsz, device, multiprocessing=True):
        self.indices = [KNN(dim) for _ in range(bsz)]
        self.bsz = bsz
        self.device = device

        self.n_jobs = cpu_count() if multiprocessing else 1

    def reset(self):
        for i in range(self.bsz):
            self.indices[i].reset()

    def add(self, queries):

        # @delayed
        # def knn_add(knn, x):
        #     knn.add(x)
        #     return knn

        # updated_knns = Parallel(n_jobs = self.n_jobs)(knn_add(*args) for args in zip(self.indices, queries))
        # for i in range(self.bsz):
        #     self.indices[i] = updated_knns[i]

        for i in range(self.bsz):
            self.indices[i].add(queries[i])

    def search(self, queries, topk):

        # @delayed
        # def knn_search(knn, query):
        #     score, value, vector = knn.search_and_reconstruct(query, topk)
        #     return torch.from_numpy(vector)

        # vectors = Parallel(n_jobs=self.n_jobs)(knn_search(*args) for args in zip(self.indices, queries))
        # vectors = torch.stack(vectors).to(self.device)

        # return vectors

        vectors, masks = [], []
        for i in range(self.bsz):
            score, value, vector = self.indices[i].search_and_reconstruct(queries[i], topk)
            vectors.append(torch.from_numpy(vector))
            # masks.append(mask)

        vectors = torch.stack(vectors).to(self.device)
        masks = torch.ones_like(vectors)

        return vectors  # , masks


class KNN:
    """
    KNN Index for one sample in minibatch
    Implements reset, train, add, and search
    Search function returns indices of topk

    scores mean distance
    values mean indices
    vectors mean hidden states
    """

    def __init__(self,
                 dim,
                 M=15
                 ):

        self.dim = dim
        self.keys = []

        # what is the M for?
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.reset()

        # xb = np.random.random((1000, dim)).astype('float32')
        # self.index.train(xb)
        # self.index.add(xb)

    def reset(self):
        self.index.reset()
        self.is_trained = False

    def train(self, x):
        self.index.train(x)
        self.is_trained = True

    def add(self, x):
        x = x.cpu().detach()

        if not self.is_trained:
            self.train(x)

        self.index.add(x)

    def search_and_reconstruct(self, queries, k):
        """how do u even perform more than topk=1? how does that even work?"""
        seqlen, dim = queries.shape
        scores, values, vectors = self.index.search_and_reconstruct(queries.cpu().detach(), k)
        vectors = vectors.reshape(seqlen, dim)

        return scores, values, vectors
