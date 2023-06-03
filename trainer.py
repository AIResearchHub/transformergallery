

import torch
import torch.nn as nn
from torch.optim import Adam


class Trainer:

    def __init__(self,
                 model,
                 memory,
                 lr,
                 batch_size,
                 n_accumulate,
                 burnin,
                 rollout,
                 ):

        self.model = nn.DataParallel(model).cuda()
        self.opt = Adam(self.model.parameters(), lr=lr)

        self.memory = memory
        self.batch_size = batch_size
        self.n_accumulate = n_accumulate

        self.burnin = burnin
        self.rollout = rollout
        self.length = burnin + rollout

    def step(self):
        """TODO: accumulating gradients"""

        X, Y, states, idxs = self.memory.get_batch(batch_size=self.batch_size)

    def get_grad(self, X, Y, state, idxs):
        self.model.zero_grad()

        expected, _ = self.model(X, state)
        loss = self.bert_loss(Y, expected)
        loss.backward()

        return loss.item()

    def bert_loss(self, target, expected):
        """
        :param target:   [batch_size, max_len]
        :param expected: [batch_size, max_len, vocab_size]
        """
        expected = expected.transpose(1, 2)

        return self.criterion(expected, target)

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    @staticmethod
    def hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

