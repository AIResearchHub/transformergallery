import faiss
import faiss.contrib.torch_utils
import time
import logging

import torch
import numpy as np

code_size = 64


class DatastoreBatch():
    def __init__(self, dim, batch_size, flat_index=False, gpu_index=False, verbose=False) -> None:
        self.indices = []
        self.batch_size = batch_size
        for i in range(batch_size):
            self.indices.append(Datastore(dim, use_flat_index=flat_index, gpu_index=gpu_index, verbose=verbose))

    def move_to_gpu(self):
        for i in range(self.batch_size):
            self.indices[i].move_to_gpu()

    def add_keys(self, keys, num_keys_to_add_at_a_time=100000):
        for i in range(self.batch_size):
            self.indices[i].add_keys(keys[i], num_keys_to_add_at_a_time)

    def train_index(self):
        for index in self.indices:
            index.train_index()

    def search(self, queries, k):
        found_scores, found_values = [], []
        for i in range(self.batch_size):
            scores, values = self.indices[i].search(queries[i], k)
            found_scores.append(scores)
            found_values.append(values)
        return torch.stack(found_scores, dim=0), torch.stack(found_values, dim=0)

    def search_and_reconstruct(self, queries, k):
        found_scores, found_values = [], []
        found_vectors = []
        for i in range(self.batch_size):
            scores, values, vectors = self.indices[i].search_and_reconstruct(queries[i], k)
            found_scores.append(scores)
            found_values.append(values)
            found_vectors.append(vectors)
        return torch.stack(found_scores, dim=0), torch.stack(found_values, dim=0), torch.stack(found_vectors, dim=0)


class Datastore:

    def __init__(self, dim):
        self.dim = dim
        self.keys = []

        self.index = faiss.IndexFlatIP(self.dim)
        self.index_size = 0

    def train_index(self):
        self.keys = torch.cat(self.keys, axis=0)

        ncentroids = int(self.keys.shape[0] / 128)
        self.index = faiss.IndexIVFPQ(self.index, self.dimension,
                                      ncentroids, code_size, 8)
        self.index.nprobe = min(32, ncentroids)

        self.index.train(self.keys)
        self.add_keys(keys=self.keys, index_is_trained=True)

    def add_keys(self, keys):
        self.keys.append(keys)

    def search_and_reconstruct(self, queries, k):
        scores, values, vectors = self.index.index.search_and_reconstruct(queries.cpu().detach(), k)
        return scores, values, vectors

    def search(self, queries, k):
        scores, values = self.index.search(queries, k)
        values[values == -1] = 0
        return scores, values


