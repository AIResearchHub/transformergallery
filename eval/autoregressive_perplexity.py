

import torch
from torcheval.metrics.text import Perplexity


def test_perplexity(model, dataloader, device):
    """
    Args:
        model:
        dataloader:

    Returns:

    """
    metric = Perplexity(device=device)

    for i, batch in enumerate(dataloader):
        bsz, timesteps, seqlen = batch.shape
        inputs, targets = batch[:, :, :-1], batch[:, :, 1:]

        with torch.no_grad():
            model.module.reset()
            for t in range(timesteps):
                expected = model(inputs[:, t, :])
                metric.update(expected, targets[:, t, :])

    ppl = metric.compute()
    print(ppl.item())

