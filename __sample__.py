

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast


def main(prompt="once upon a time", num_samples=10, device="cuda"):
    model = torch.load("saved/final").to(device)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # (bsz, seqlen)
    x = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
    x = torch.tensor(x, dtype=torch.int64, device=device).repeat(num_samples, 1)

    model.module.reset()
    for _ in range(500):
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # x    (bsz, seqlen)
        # pred (bsz, 1)
        x = torch.concat((x, idx_next), dim=-1)

    for sample in x:
        sentence = tokenizer.batch_decode(sample)
        print(' '.join(sentence))


if __name__ == "__main__":
    main()

