
import torch
from torcheval.metrics.text import Perplexity


def test_perplexity_sep(model, dataloader, device, sep_token=102, vocab_size=30522):
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
        seqlen -= 1

        with torch.no_grad():
            model.module.reset()
            for t in range(timesteps):
                expected = model(inputs[:, t, :])
                # expected = (bsz, seq_len, vocab_size)

                expected = expected.reshape(-1, vocab_size)
                targets_ = targets[:, t, :].reshape(-1)

                mask = (inputs[:, t, :].reshape(-1) != sep_token) & \
                       (inputs[:, t, :].reshape(-1) != 50) & \
                       (inputs[:, t, :].reshape(-1) != 51)

                assert targets_.shape == (bsz * seqlen,)
                assert expected.shape == (bsz * seqlen, vocab_size)

                targets_ = targets_[mask]
                expected = expected[mask]

                metric.update(expected.unsqueeze(0), targets_.unsqueeze(0))

    ppl = metric.compute()
    return ppl.item()

