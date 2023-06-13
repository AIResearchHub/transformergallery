

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

    def __init__(self, d_model, n_head, bsz, topk):
        super(KNNAttention, self).__init__()
        self.n_head = n_head
        self.topk = topk

        # self.memory = KNNMemory(bsz)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def reset(self):
        self.memory.reset()

    def forward(self, q, kv, mask=None):
        """
        get queries to compute, and keys and values hidden states to store
        in the knn memory

        Parameters:
            q (bsz, seqlen, dim): hidden states to compute queries
            kv (bsz, seqlen, dim): hidden states to store in knn memory
        """
        q = self.w_q(q)

        # knn memory
        # mem, mem_mask = self.memory.search(q, topk=self.topk)
        # self.memory.add(kv)

        # k, v = self.w_kv(mem).chunk(2, dim=-1)
        k, v = self.w_kv(kv).chunk(2, dim=-1)

        q, k, v = rearrange(q, 'b l (h d) -> b h l d', h=self.n_head), k.unsqueeze(1), v.unsqueeze(1)

        # attention reformulation here?

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.w_concat(out)

        return out


class KNNMemory:

    def __init__(self, bsz, shape, memmap_filename, multiprocessing=True):

        self.db = np.memmap(memmap_filename, mode='w+', dtype=np.float32, shape=shape)
        self.knns = [KNN for _ in range(bsz)]

        self.n_jobs = cpu_count() if multiprocessing else 1

    def reset(self):
        pass

    def add(self):
        pass

    def search(self,
               queries,
               topk,
               nprobe=8):
        """
        what is nprobe?

        """
        device = queries.device
        queries = queries.detach().cpu().numpy()

        all_states = []
        all_masks = []

        # perform KNN searches in parallel across batch
        knns = [self.knns[i] for i in self.scoped_indices]

        @delayed
        def knn_search(knn, query):
            return knn.search(query, topk, nprobe)

        fetched_indices = Parallel(n_jobs=self.n_jobs)(knn_search(*args) for args in zip(knns, queries))

        for batch_index, indices in zip(self.scoped_indices, fetched_indices):

            # false if indices is -1
            mask = (indices != -1)
            # convert all the -1 indices to 0
            db_indices = np.where(mask, indices, 0)
            # get values from memory after accounting for max memories
            states = self.db[batch_index, db_indices % self.max_memories]

            all_states.append(torch.from_numpy(states))
            all_masks.append(torch.from_numpy(mask))

        all_states = torch.stack(all_states)
        all_masks = torch.stack(all_masks)

        return all_states.to(device), all_masks.to(device)


class KNN:
    """
    KNN Index for one sample in minibatch
    Implements reset, train, add, and search
    Search function returns indices of topk
    """

    def __init__(self,
                 dim,
                 M
                 ):
        # faiss index with inner product?

        self.index = faiss.IndexHSNWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.is_trained = False

        self.reset()

    def reset(self):
        self.index.reset()
        self.is_trained = False

    def train(self, x):
        self.index.train(x)
        self.is_trained = True

    def add(self, x):
        if not self.is_trained:
            self.train(x)

        return self.index.add(x)

    def search(self, x, topk):
        if not self.is_trained:
            return np.full((x.shape[0], topk), -1)

        distances, indices = self.index.search(x, k=topk)

        return indices

