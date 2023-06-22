

from torch.utils.data import DataLoader

from transformer import *
from autoregressivetrainer import AutoregressiveTrainer
from berttrainer import BertTrainer
from dataset import PG19Dataset


def main(seq_len=512,
         vocab_size=30522,
         n_layers=4,
         d_model=768,
         n_head=8,
         p=0.1,
         lr=4e-5,
         batch_size=16,
         burnin=0,
         rollout=1,
         warmup_steps=100,
         device="cuda",
         cache_dir="/media/yh04/New Volume/datasets"
         ):

    lm = AutoregressiveLM(
        cls=Transformer,
        vocab_size=vocab_size,
        max_len=seq_len,
        n_layers=n_layers,
        d_model=d_model,
        n_head=n_head,
        p=p,
        device=device,
        w=512,
        bsz=batch_size,
        topk=1,
    )
    lm.load_pretrained()

    dataloader = DataLoader(
        PG19Dataset(
            cache_dir=cache_dir,
            split="train[:2000]",
            seq_len=seq_len + 1,
            block_len=rollout,
            device=device
        ),
        batch_size=batch_size,
    )

    trainer = AutoregressiveTrainer(
        model=lm,
        dataloader=dataloader,
        lr=lr,
        batch_size=batch_size,
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

