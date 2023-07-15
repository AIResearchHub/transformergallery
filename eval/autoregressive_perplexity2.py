

import torch
from torcheval.metrics.text import Perplexity


def test_perplexity(model, dataloader, device, w=128, sep_token=102, vocab_size=30522):
    """
    Args:
        model:
        dataloader: dataloader that has sep padding

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
                # expected = (bsz, seq_len, vocab_size)

                expected = expected.view(-1, vocab_size)
                targets_ = targets[:, t, :].view(-1)

                mask = (inputs[:, t, :].view(-1) != sep_token)
                expected = expected[mask]
                targets_ = targets_[mask]

                assert targets_.shape == (bsz * seqlen)
                assert expected.shape == (bsz * seqlen, vocab_size)

                metric.update(expected, targets_)

    ppl = metric.compute()
    print(ppl.item())

