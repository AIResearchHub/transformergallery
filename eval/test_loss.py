

import torch.nn.functional as F

from ..utils import apply_mlm_mask


def test_loss(model, dataloader):
    for i, data in enumerate(dataloader):
        inputs, target = apply_mlm_mask(data)
        pred = model(inputs)

        expected = expected.transpose(1, 2)

        loss = self.criterion(expected, target)
        return loss.mean()

        F.nll_loss(pred.transpose(1, 2), target)
