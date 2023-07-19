

import torch
from torch.utils.data import DataLoader

from dataset import TextDataset
from eval import test_loss, test_memory, test_perplexity, test_perplexity_sep, test_reasoning


def main(cache_dir="/media/yh04/New Volume/datasets",
         device="cuda"):

    model = torch.load("saved/arxivrecsep120000ppl23").to(device)

    dataloader = DataLoader(
        TextDataset(
            name="scientific_papers",
            cache_dir=cache_dir,
            split="test",
            seq_len=512,
            block_len=5,
            device=device,
            sep_padding=False,
            # max_len=10
        ),
        batch_size=8,
    )

    ppl = test_perplexity_sep(model, dataloader, device)
    print("No [SEP] Perplexity: ", ppl)

    dataloader = DataLoader(
        TextDataset(
            name="scientific_papers",
            cache_dir=cache_dir,
            split="test",
            seq_len=512,
            block_len=5,
            device=device,
            sep_padding=True,
            # max_len=10
        ),
        batch_size=8,
    )
    ppl = test_perplexity_sep(model, dataloader, device)
    print("[SEP] Perplexity: ", ppl)


if __name__ == "__main__":
    main()

