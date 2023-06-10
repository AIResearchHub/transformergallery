

import torch
import torch.nn as nn
from torch.optim import Adam

import time

from optim_schedule import ScheduledOptim


def apply_mlm_mask(batch, mask_prob):
    device = batch.device

    probs = torch.rand(*batch.shape)
    masks = (probs < mask_prob).to(device)

    # create inputs
    inputs = batch.detach() * torch.logical_not(masks).to(device)
    inputs[inputs == 0] = 103

    # create labels
    labels = batch.detach() * masks

    return inputs.long(), labels.long()


class Trainer:

    def __init__(self,
                 model,
                 dataloader,
                 lr,
                 batch_size,
                 n_accumulate,
                 burnin,
                 rollout,
                 warmup_steps,
                 device,
                 ):

        self.model = nn.DataParallel(model)
        if device == "cuda":
            self.model = self.model.cuda()
        self.opt = Adam(self.model.parameters(), lr=lr)
        self.opt_schedule = ScheduledOptim(self.opt, self.model.module.d_model, n_warmup_steps=warmup_steps)

        self.criterion = nn.NLLLoss(ignore_index=0)

        self.dataloader = dataloader
        self.batch_size = batch_size
        self.n_accumulate = n_accumulate

        self.burnin = burnin
        self.rollout = rollout
        self.length = burnin + rollout

        self.log = open(f"logs/lm", "w")
        self.start = time.time()
        self.updates = 0

    def run_epoch(self, epoch):

        start = time.time()
        for i, batch in enumerate(self.dataloader):
            # batch (bsz, block_len, seq_len)
            loss = self.step(batch)

            self.log.write(f"{time.time() - self.start}, {loss}\n")
            self.log.flush()

            self.updates += 1

            if i % 5 == 0:
                print(f"Epoch: {epoch} \t "
                      f"Time: {time.time() - self.start} \t "
                      f"Loss: {loss} \t "
                      f"Update/Sec: {(time.time() - self.start) / self.updates}")

            if i % 10000 == 0:
                torch.save(self.model, "saved/final")

    def step(self, batch):
        total_loss = 0
        inputs, targets = apply_mlm_mask(batch, mask_prob=0.25)

        state = None
        for t in range(self.rollout):
            expected, state = self.model(inputs[:, t, :], state=state)
            loss = self.bert_loss(expected, targets[:, t, :])
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.opt.step()

            total_loss += loss.item()

        return total_loss

    def bert_loss(self, expected, target):
        """
        negative log likelihood takes in log probabilities and labels
        and outputs loss

        Parameters:
            expected (batch_size, max_len, vocab_size)
            target (batch_size, max_len)
        """
        assert target.shape == (target.size(0), target.size(1))
        assert expected.shape == (target.size(0), target.size(1), expected.size(2))

        expected = expected.transpose(1, 2)

        return self.criterion(expected, target)
