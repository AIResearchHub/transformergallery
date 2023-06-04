

import torch
import time

from transformer import TransformerLM, Transformer, TransformerXL
from memory import Memory
from trainer import Trainer
from dataset import create_bert_data


def main():
    data = create_bert_data(max_files=1000)

    lm = TransformerLM(
        cls=Transformer,
        vocab_size=30522,
        n_layers=4,
        d_model=512,
        n_head=8,
        p=0.1
    )

    mem = Memory(
        data=data,
        init=lm.init_state()
    )

    trainer = Trainer(
        model=lm,
        memory=mem,
        lr=1e-4,
        batch_size=32,
        n_accumulate=2,
        burnin=0,
        rollout=1
    )

    filename = f"logs/lm"
    log = open(filename, "w")

    timesteps = 100000
    start = time.time()
    for i in range(timesteps):
        loss = trainer.step()

        print(f"{time.time() - start}, {loss}")
        log.write(f"{time.time() - start}, {loss}\n")
        log.flush()

        if i % 100 == 0:
            torch.save(trainer.model, "saved/final")


if __name__ == "__main__":
    main()

