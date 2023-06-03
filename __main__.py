

import torch

from transformer import TransformerLM, Transformer, TransformerXL
from memory import Memory
from trainer import Trainer
from dataset import create_bert_data


def main():
    data = create_bert_data(max_files=10)

    lm = TransformerLM(
        TransformerXL,
        vocab_size=30522,
        n_layers=4,
        d_model=512,
        n_head=8,
        p=0.1
    )

    mem = Memory(
        data=data
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
    for i in range(timesteps):
        loss = trainer.step()

        print(f"{loss}")
        log.write(f"{loss}\n")
        log.flush()

        if i % 100 == 0:
            torch.save(trainer.model, "models/final")


if __name__ == "__main__":
    main()

