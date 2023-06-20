

import torch
from torch.utils.data import DataLoader

from transformer import TransformerLM, Transformer
from dataset import PG19Dataset
from eval import test_loss, test_memory, test_reasoning


def main(seq_len=512,
         vocab_size=30522,
         n_layers=4,
         d_model=768,
         n_head=8,
         p=0.1,
         batch_size=32,
         rollout=1,
         device="cuda",
         cache_dir="/media/yh04/New Volume/datasets"
         ):
    model = TransformerLM(
        cls=Transformer,
        vocab_size=vocab_size,
        max_len=seq_len,
        n_layers=n_layers,
        d_model=d_model,
        n_head=n_head,
        p=p,
        device="cpu",
        w=512,
        bsz=1,
        topk=1,
    )
    model.load_state_dict(torch.load("saved/final"))

    dataloader = DataLoader(
        PG19Dataset(
            cache_dir=cache_dir,
            split="validation",
            seq_len=512,
            block_len=rollout,
            device=device
        ),
        batch_size=batch_size,
    )

    loss = test_loss(model, dataloader)
    print(loss)


if __name__ == "__main__":
    main()

