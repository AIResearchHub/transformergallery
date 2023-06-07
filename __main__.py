

import torch
import time

from transformer import *
from memory import Memory
from trainer import Trainer
from dataset import create_bert_data


def main(max_files=1000,
         max_len=1024,
         vocab_size=30522,
         n_layers=4,
         d_model=512,
         n_head=8,
         p=0.1,
         lr=1e-4,
         batch_size=16,
         n_accumulate=1,
         burnin=0,
         rollout=5
         ):

    data = create_bert_data(
        max_files=max_files,
        max_len=max_len
    )

    lm = TransformerLM(
        cls=LongformerXL,
        vocab_size=vocab_size,
        max_len=max_len,
        n_layers=n_layers,
        d_model=d_model,
        n_head=n_head,
        p=p
    )

    mem = Memory(
        data=data,
        init=lm.init_state(),
        max_len=max_len,
        n_layers=n_layers,
        d_model=d_model
    )

    trainer = Trainer(
        model=lm,
        memory=mem,
        lr=lr,
        batch_size=batch_size,
        n_accumulate=n_accumulate,
        burnin=burnin,
        rollout=rollout
    )

    filename = f"logs/lm"
    log = open(filename, "w")

    timesteps = 100000
    start = time.time()
    for i in range(timesteps):
        loss = trainer.step()

        print(f"Time: {time.time() - start} \t Loss: {loss} \t Update/Sec: {(time.time() - start) / (i + 1)}")
        log.write(f"{time.time() - start}, {loss}\n")
        log.flush()

        if i % 100 == 0:
            torch.save(trainer.model, "saved/final")


if __name__ == "__main__":
    main()

