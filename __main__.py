import argparse
from torch.utils.data import DataLoader

from transformer import *
from autoregressivetrainer import AutoregressiveTrainer
from berttrainer import BertTrainer
from dataset import TextDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", default=512, type=int)
    parser.add_argument("--vocab_size", default=30522, type=int)
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--d_model", default=768, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--p", default=0.1, type=float)
    parser.add_argument("--lr", default=4e-5, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--burnin", default=0, type=int)
    parser.add_argument("--rollout", default=5, type=int)
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--cache_dir", default="./cache/datasets", type=str)

    args = parser.parse_args()

    lm = AutoregressiveLM(
        cls=RecurrentMemoryTransformer,
        vocab_size=args.vocab_size,
        max_len=args.seq_len,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_head=args.n_head,
        p=args.p,
        device=args.device,
        w=128,
        bsz=args.batch_size,
        topk=1,
        num_tokens=128,
        mem_tokens=64,
    )
    lm.load_pretrained()

    dataloader = DataLoader(
        TextDataset(
            name="scientific_papers",
            cache_dir=args.cache_dir,
            split="train[:20000]",
            seq_len=args.seq_len,
            block_len=args.rollout,
            device=args.device,
            sep_padding=True
        ),
        batch_size=args.batch_size,
    )

    trainer = AutoregressiveTrainer(
        model=lm,
        dataloader=dataloader,
        lr=args.lr,
        batch_size=args.batch_size,
        seqlen=args.seq_len,
        burnin=args.burnin,
        rollout=args.rollout,
        warmup_steps=args.warmup_steps,
        device=args.device
    )
    print("Starting training run...")

    epochs = 1000
    for epoch in range(epochs):
        trainer.run_epoch(epoch)


if __name__ == "__main__":
    main()
