

import torch
import torch.nn.functional as F


def apply_memory_mask(batch, init_len=3):
    device = batch.device

    memory_tokens = batch[:, :init_len, :].detach()
    masks = torch.zeros(*batch.shape)
    for token in memory_tokens.squeeze():
        masks[masks == token] = 1

    masks = masks.to(device)

    # create inputs
    inputs = batch.detach() * torch.logical_not(masks).to(device)
    inputs[inputs == 0] = 103

    # create labels
    labels = batch.detach() * masks

    # recover initial token lengths
    inputs[:, :init_len, :] = memory_tokens
    labels[:, :init_len, :] = 0

    return inputs.long(), labels.long()


def test_memory(model, dataloader):
    """
    mask out all the titles, names and locations in pg19 to see if
    model with memory can better predict those tokens

    Parameters:
        model (nn.Module): The model to be evaluated
        dataloader (Dataloader): Batch size 1 dataloader with entire book length and shape (timesteps, seqlen)
    """
    for i, batch in enumerate(dataloader):
        bsz, timesteps, seqlen = batch.shape

        inputs, targets = apply_memory_mask(batch)
        total_loss = 0

        with torch.no_grad():
            model.module.reset()
            for t in range(timesteps):
                expected = model(inputs[:, t, :])
                loss = F.nll_loss(expected.transpose(1, 2), targets[:, t, :])
                total_loss += loss.item()

        print(total_loss / (timesteps * seqlen * bsz))

