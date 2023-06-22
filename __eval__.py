

import torch
from torch.utils.data import DataLoader

from transformer import TransformerLM, Transformer
from dataset import PG19Dataset
from eval import test_loss, test_memory, test_reasoning


def main(cache_dir="/media/yh04/New Volume/datasets"
         ):

    model = torch.load("saved/final").cpu()

    dataloader = DataLoader(
        PG19Dataset(
            cache_dir=cache_dir,
            split="validation",
            seq_len=512,
            block_len=1,
            device="cpu"
        ),
        batch_size=1,
    )

    # loss = test_loss(model, dataloader)
    loss = test_memory(model, dataloader)
    print(loss)


if __name__ == "__main__":
    main()

