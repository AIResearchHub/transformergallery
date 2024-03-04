

import torch
from transformers import BertTokenizerFast

from utils import generate_samples, join, remove_padding


def main(prompt="", num_samples=8, seq_len=512, device="cuda:0"):
    model = torch.load("saved/arxivrecsep120000ppl23").to(device)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    prompt_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))

    samples = generate_samples(model, prompt_ids=prompt_ids, seq_len=seq_len,
                               B=num_samples, T=1024-2, temperature=1.0)

    for sample in samples:
        sample, mask = remove_padding(sample)
        print("SAMPLE:")
        print()
        sentence = tokenizer.batch_decode(sample)
        print(join(sentence))


if __name__ == "__main__":
    main()

