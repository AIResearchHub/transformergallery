

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

        loss = self.get_grad(X, Y, states, idxs)
        self.opt.step()

        return loss

    def get_grad(self, X, Y, state, idxs):
        X = X[0]
        Y = Y[0]
        print(len(state))

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

