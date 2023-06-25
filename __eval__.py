

import torch
from torch.utils.data import DataLoader

from dataset import PG19Dataset
from eval import test_loss, test_memory, test_perplexity, test_reasoning


def main(cache_dir="/media/yh04/New Volume/datasets",
         device="cuda"):

    model = torch.load("saved/final").to(device)

    dataloader = DataLoader(
        PG19Dataset(
            cache_dir=cache_dir,
            split="validation",
            seq_len=512 + 1,
            block_len=5,
            device=device
        ),
        batch_size=1,
    )

    # loss = test_loss(model, dataloader)
    # loss = test_memory(model, dataloader)
    loss = test_perplexity(model, dataloader, device)
    print(loss)


if __name__ == "__main__":
    main()

