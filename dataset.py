

import torch
from torch.utils.data import Dataset

import random

from utils import create_pg19_data


class PG19Dataset(Dataset):
    """
    data = List of torch tensors with (block_len, seq_len)
    seq_len = sequence length to be fed into model per timestep e.g. 2 = (2, 512)
    device = cuda or cpu
    """

    def __init__(self, max_files, seq_len, block_len, device):
        super().__init__()
        #  [(total_len, seq_len), ...]
        self.data = create_pg19_data(path="data/pg19/train", max_len=seq_len, max_files=max_files)
        self.seq_len = seq_len
        self.block_len = block_len
        self.device = device

        self.size = sum([x.size(0) for x in self.data]) // block_len

    def __getitem__(self, index):
        bidx = random.randrange(0, len(self.data))
        tidx = random.randrange(0, self.data[bidx].size(0) - self.block_len + 1)
        data = self.data[bidx][tidx:tidx+self.block_len].long()
        return data.to(self.device)

    def __len__(self):
        return self.size

