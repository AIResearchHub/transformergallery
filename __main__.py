

from torch.utils.data import DataLoader

from transformer import *
from trainer import Trainer
from dataset import PG19Dataset


def main(seq_len=4096,
         vocab_size=30522,
         n_layers=4,
         d_model=768,
         n_head=8,
         p=0.1,
         lr=1e-4,
         batch_size=1,
         n_accumulate=1,
         burnin=0,
         rollout=1,
         warmup_steps=100,
         device="cuda",
         cache_dir="/media/yh04/New Volume/datasets"
         ):

    dataloader = DataLoader(
        PG19Dataset(
            cache_dir=cache_dir,
            split="validation",
            seq_len=seq_len,
            block_len=rollout,
            device=device
        ),
        batch_size=batch_size,
    )

    lm = TransformerLM(
        cls=LongformerHuggingface,
        vocab_size=vocab_size,
        max_len=seq_len,
        n_layers=n_layers,
        d_model=d_model,
        n_head=n_head,
        p=p,
        device=device,
        w=512,
    )

    trainer = Trainer(
        model=lm,
        dataloader=dataloader,
        lr=lr,
        batch_size=batch_size,
        n_accumulate=n_accumulate,
        seqlen=seq_len,
        burnin=burnin,
        rollout=rollout,
        warmup_steps=warmup_steps,
        device=device
    )

    epochs = 1000
    for epoch in range(epochs):
        trainer.run_epoch(epoch)


if __name__ == "__main__":
    main()

