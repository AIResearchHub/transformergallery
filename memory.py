

import torch
import random


class Memory:

    def __init__(self, data):
        self.data = data
        self.size = len(data)

        self.state = [[None * seq.size(0)] for seq in data]

    def update_state(self, idxs, t, states):
        for idx, state in zip(idxs, states):
            self.state[idx[0]][idx[1]+t] = state

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
            timeidx = random.randrange(0, self.data[bufferidx].size(0)-length+1)
            idxs.append([bufferidx, timeidx])

            tokens = self.data[bufferidx][timeidx:timeidx+length]
            x, y = self.mask_tokens(tokens, p=0.25)
            X.append(x)
            Y.append(y)

            states.append(self.state[bufferidx][timeidx])

        X = torch.stack(X).to("cuda")
        Y = torch.stack(Y).to("cuda")
        states = torch.stack(states).to("cuda")

        X = X.transpose(0, 1)
        Y = Y.transpose(0, 1)

        assert X.shape == (length, batch_size, X.size(2))
        assert Y.shape == (length, batch_size, Y.size(2))
        assert states.shape == (batch_size, self.dim)

        return X, Y, states, idxs

