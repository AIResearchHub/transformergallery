

import torch.nn.functional as F

from utils import apply_mlm_mask


def test_loss(model, dataloader):
    for i, batch in enumerate(dataloader):
        timesteps, seqlen = batch.shape
        inputs, targets = apply_mlm_mask(batch)

        total_loss = 0

        model.module.reset()
        for t in range(timesteps):
            expected = model(inputs[:, t, :])
            loss = F.nll_loss(expected.transpose(1, 2), targets[:, t, :])
            total_loss += loss.item()

        return total_loss / (timesteps * seqlen)

