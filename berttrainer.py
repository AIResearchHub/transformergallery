

import torch
import torch.nn as nn
from torch.optim import AdamW

import time
import datetime

from utils import apply_mlm_mask


class BertTrainer:
    """
    Trains a Large Language Model to predict the masked words in the inputs.
    """

    print_every = 5
    save_every = 1000

    def __init__(self,
                 model,
                 dataloader,
                 lr,
                 batch_size,
                 accum,
                 seqlen,
                 burnin,
                 rollout,
                 device="cuda"
                 ):

        self.model = nn.DataParallel(model).to(device)
        self.opt = AdamW(self.model.parameters(), lr=lr)

        self.criterion = nn.NLLLoss(ignore_index=0)

        self.dataloader = dataloader
        self.batch_size = batch_size
        self.accum = accum

        self.seqlen = seqlen
        self.burnin = burnin
        self.rollout = rollout
        self.length = burnin + rollout

        self.dt = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
        self.log = open(f"logs/{self.dt}", "w")
        self.start = time.time()
        self.updates = 0

    def run_epoch(self, epoch):
        """
        Main training loop for one epoch.
        Logs and prints time and loss.
        """

        for i, batch in enumerate(self.dataloader):
            if batch.size(0) != self.batch_size:
                continue

            # (B, T, S)
            loss = self.step(i, batch)

            self.log.write(f"{time.time() - self.start}, {loss}\n")
            self.log.flush()

            self.updates += 1

            if i % self.print_every == 0:
                print(f"Epoch: {epoch} \t "
                      f"Time: {time.time() - self.start} \t "
                      f"Loss: {loss} \t "
                      f"Sec/Update: {(time.time() - self.start) / self.updates}")

            if i % self.save_every == 0:
                torch.save(self.model, "saved/final")

    def step(self, i, batch):
        """
        A training step that does backpropagation at each rollout timestep.
        To train long sequence transformer models such as Transformer XL.
        """
        total_loss = 0
        inputs, targets = apply_mlm_mask(batch, mask_prob=0.25)

        self.model.module.reset()
        for t in range(self.rollout):
            expected = self.model(inputs[:, t, :])
            total_loss += self.bert_loss(expected, targets[:, t, :])

        total_loss = total_loss / self.accum
        total_loss.backward()

        if i % self.accum == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.opt.step()
            self.opt.zero_grad()

        return total_loss.item() / (self.rollout * self.seqlen)

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

        loss = self.criterion(expected, target)
        return loss.mean()
