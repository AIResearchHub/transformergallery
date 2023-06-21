

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
    Instead of standard attention with computed query,
    keys, and values, the query is searched against cached
    keys and values to incorporate past information.
    A few tricks are used from the paper such as
    normalizing keys and values to prevent distribution shift,
    even if the weights has changed, the magnitude of the keys and
    values remain the same because of normalization.

    Parallel code taken from lucidrains memorizing transformer
    however it runs slower than for loop.

    """

    def __init__(self, d_model, n_head, bsz, device):
        super(KNNAttention, self).__init__()
        self.n_head = n_head
        self.bsz = bsz

        self.memory = KNNMemory(d_model // n_head, bsz, device)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)

        # memorizing transformer gate
        self.bias = nn.Parameter(torch.randn(d_model // n_head), requires_grad=True)

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
        q, k, v = self.w_q(q), *self.w_kv(kv).chunk(2, dim=-1)
        k, v = F.normalize(k), F.normalize(v)

        self.memory.add(torch.stack((k, v), dim=-2))

        # perform local attention
        q, k, v = rearrange(q, 'b l (h d) -> b h l d', h=self.n_head), k.unsqueeze(1), v.unsqueeze(1)
        local_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # get cached keys and values from memory
        mem = self.memory.search(q, topk=topk)
        k, v = mem.unbind(dim=-2)

        # perform external memory attention
        retrieved_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # is the sigmoid needed?
        gate = F.sigmoid(self.bias)
        out = retrieved_attn * gate + local_attn * (1 - gate)

        # out = local_attn + retrieved_attn

        out = rearrange(out, 'b h l d -> b l (h d)', h=self.n_head)
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
        print("Number of jobs: ", self.n_jobs)

    def reset(self):
        for i in range(self.bsz):
            self.indices[i].reset()

    def add(self, kv):
        kv = kv.cpu().detach().numpy()

        # @delayed
        # def knn_add(knn, x):
        #     knn.add(x)
        #     return knn
        #
        # updated_knns = Parallel(n_jobs=self.n_jobs)(knn_add(*args) for args in zip(self.indices, kv))
        # for i in range(self.bsz):
        #     self.indices[i] = updated_knns[i]

        for i in range(self.bsz):
            self.indices[i].add(kv[i])

    def search(self, queries, topk):
        """TODO: add indices masks"""
        bsz, nhead, seqlen, dhead = queries.shape
        queries = queries.reshape(bsz, -1, dhead)
        queries = queries.cpu().detach().numpy()

        # @delayed
        # def knn_search(knn, query):
        #     score, value, vector = knn.search_and_reconstruct(query, topk)
        #     return torch.from_numpy(vector)
        #
        # vectors = Parallel(n_jobs=self.n_jobs)(knn_search(*args) for args in zip(self.indices, queries))
        # vectors = torch.stack(vectors).to(self.device)
        #
        # return vectors

        vectors, masks = [], []
        for i in range(self.bsz):
            vector = self.indices[i].search(queries[i], topk)
            vectors.append(torch.from_numpy(vector))

        vectors = torch.stack(vectors).to(self.device)
        vectors = vectors.reshape(bsz, nhead, seqlen, 2, dhead)

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

        self.db = []
        # self.db = np.zeros(())

        # what is the M for?
        # approximate kNN memory
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.reset()

        # xb = np.random.random((1000, dim)).astype('float32')
        # self.index.train(xb)
        # self.index.add(xb)

    def reset(self):
        self.db = []
        self.index.reset()
        self.is_trained = False

    def train(self, x):
        self.index.train(x)
        self.is_trained = True

    def add(self, x):
        # if not self.is_trained:
        #     self.train(x)
        keys = np.ascontiguousarray(x[..., 0, :])

        # add to memory
        for _x in x:
            self.db.append(_x)

        self.index.add(keys)

    def search(self, queries, k):
        distance, indices = self.index.search(queries, k)

        vectors = np.array([self.db[i[0]] for i in indices])
        return vectors

    def search_and_reconstruct(self, queries, k):
        """how do u even perform more than topk=1? how does that even work?"""
        seqlen, dim = queries.shape
        scores, values, vectors = self.index.search_and_reconstruct(queries, k)
        vectors = vectors.reshape(seqlen, dim)

        return scores, values, vectors
