

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast


def join(strings):
    output = ""
    for string in strings:
        if string[:2] == "##":
            output += string[2:]
        elif string == "i" or string == "a":
            output += ' ' + string
        elif len(string) == 1:
            output += string
        else:
            output += ' ' + string

    return output


def main(prompt="therefore, the equation can be written as", num_samples=1, device="cuda"):
    model = torch.load("saved/arxivrecsep120000ppl23").to(device)
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
        print("=================================================================================")
        sentence = tokenizer.batch_decode(sample)
        print(join(sentence))


if __name__ == "__main__":
    main()

