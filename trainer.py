

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
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.memory = memory
        self.batch_size = batch_size
        self.n_accumulate = n_accumulate

        self.burnin = burnin
        self.rollout = rollout
        self.length = burnin + rollout

    def step(self):
        """TODO: accumulating gradients"""

        X, Y, states, idxs = self.memory.get_batch(batch_size=self.batch_size)

        loss = self.get_grad(X, Y, states)
        self.opt.step()

        return loss

    def get_grad(self, X, Y, state):
        X = X[0]
        Y = Y[0]

        self.model.zero_grad()
        total_loss = 0
        for i in range(self.rollout):
            expected, state = self.model(X, state)
            loss = self.bert_loss(Y, expected)
            loss.backward()
            total_loss += loss.item()

        for x in self.model.parameters():
            if x.grad is not None:
                x.grad.data.mul_(1/self.rollout)

        return total_loss

    def bert_loss(self, target, expected):
        """
        :param target:   [batch_size, max_len]
        :param expected: [batch_size, max_len, vocab_size]
        """
        assert target.shape == (self.batch_size, target.size(1))
        assert expected.shape == (self.batch_size, target.size(1), expected.size(2))

        expected = expected.transpose(1, 2)

        return self.criterion(expected, target)
