

import torch
from torch.utils.data import DataLoader
import time

from transformer import *
from trainer import Trainer
from dataset import PG19Dataset


def main(max_files=2,
         max_len=512,
         vocab_size=30522,
         n_layers=1,
         d_model=256,
         n_head=8,
         p=0.1,
         lr=1e-4,
         batch_size=2,
         n_accumulate=1,
         burnin=0,
         rollout=1,
         device="cpu"
         ):

    dataset = PG19Dataset(
        max_files=max_files,
        seq_len=max_len,
        block_len=burnin+rollout,
        device=device
    )

    dataloader = DataLoader(dataset, batch_size=batch_size)

    lm = TransformerLM(
        cls=Transformer,
        vocab_size=vocab_size,
        max_len=max_len,
        n_layers=n_layers,
        d_model=d_model,
        n_head=n_head,
        p=p,
        device=device
    )

    trainer = Trainer(
        model=lm,
        dataloader=dataloader,
        lr=lr,
        batch_size=batch_size,
        n_accumulate=n_accumulate,
        burnin=burnin,
        rollout=rollout,
        device=device
    )

    epochs = 10
    for i in range(epochs):
        trainer.run_epoch()


if __name__ == "__main__":
    main()

