

import torch
import torch.nn.functional as F

from utils import apply_mlm_mask


def test_loss(model, dataloader):
    for i, batch in enumerate(dataloader):
        bsz, timesteps, seqlen = batch.shape
        inputs, targets = apply_mlm_mask(batch, mask_prob=0.25)

        total_loss = 0

        with torch.no_grad():
            model.module.reset()
            for t in range(timesteps):
                expected = model(inputs[:, t, :])
                loss = F.nll_loss(expected.transpose(1, 2), targets[:, t, :])
                total_loss += loss.item()

        print(total_loss / (timesteps * seqlen * bsz))

