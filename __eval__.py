

import torch
from torch.utils.data import DataLoader

from dataset import PG19Dataset
from eval import test_loss, test_memory, test_perplexity, test_perplexity_sep, test_reasoning


def main(cache_dir="/media/yh04/New Volume/datasets",
         device="cuda"):

    model = torch.load("saved/recsep200000ppl60").to(device)

    dataloader = DataLoader(
        PG19Dataset(
            name="scientific_papers",
            cache_dir=cache_dir,
            split="validation",
            seq_len=512,
            block_len=5,
            device=device,
            sep_padding=False,
        ),
        batch_size=8,
    )

    ppl = test_perplexity_sep(model, dataloader, device)
    print("No [SEP] Perplexity: ", ppl)

    dataloader = DataLoader(
        PG19Dataset(
            name="scientific_papers",
            cache_dir=cache_dir,
            split="validation",
            seq_len=512,
            block_len=5,
            device=device,
            sep_padding=True,
        ),
        batch_size=8,
    )
    ppl = test_perplexity_sep(model, dataloader, device)
    print("[SEP] Perplexity: ", ppl)


if __name__ == "__main__":
    main()

