

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast


def main(prompt="hello world"):
    model = torch.load("saved/final").cuda()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # (bsz, seqlen)
    x = torch.tensor(2002, dtype=torch.int64).view(1, 1).cuda()
    for _ in range(500):
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # x    (bsz, seqlen)
        # pred (bsz, 1)
        x = torch.concat((x, idx_next), dim=-1)

    sentence = tokenizer.batch_decode(x.squeeze())
    print(' '.join(sentence))


if __name__ == "__main__":
    main()

