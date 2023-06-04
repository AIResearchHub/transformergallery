

import torch
import random

from copy import deepcopy


class Memory:

    def __init__(self, data, init):
        self.data = data
        self.size = len(data)

        self.state = [[deepcopy(init).cpu() for _ in range(seq.size(0))] for seq in self.data]

    def update_state(self, idxs, t, states):
        """states (n_layers, seq_len, batch_size, d_model)"""
        states = states.transpose(0, 2)

        for idx, state in zip(idxs, states):
            self.state[idx[0]][idx[1]+t] = state.detach().transpose(0, 1).unsqueeze(2).cpu()

    def mask_tokens(self, tokens, p):
        target = torch.zeros(*tokens.size(), dtype=torch.int64)

        for i in range(len(tokens)):
            for j in range(len(tokens[i])):
                prob = random.random()

                if prob < p:
                    target[i][j] = tokens[i][j]
                    tokens[i][j] = 103

        return tokens, target

    def get_batch(self, batch_size=32, length=1):
        X = []
        Y = []
        states = []
        idxs = []

        for i in range(batch_size):
            bufferidx = random.randrange(0, self.size)
            timeidx = random.randrange(0, self.data[bufferidx].size(0)-length)
            idxs.append([bufferidx, timeidx])

            tokens = self.data[bufferidx][timeidx:timeidx+length]
            x, y = self.mask_tokens(tokens, p=0.25)
            X.append(x)
            Y.append(y)

            states.append(self.state[bufferidx][timeidx])

        X = torch.stack(X).cuda()
        Y = torch.stack(Y).cuda()

        states = torch.concat(states, dim=2).cuda()

        X = X.transpose(0, 1)
        Y = Y.transpose(0, 1)

        assert X.shape == (length, batch_size, X.size(2))
        assert Y.shape == (length, batch_size, Y.size(2))

        return X, Y, states, idxs
