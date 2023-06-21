
from transformers import BertTokenizerFast, BartTokenizerFast

import logging
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)

import torch
import os


def apply_mlm_mask(batch, mask_prob):
    device = batch.device

    probs = torch.rand(*batch.shape)
    masks = (probs < mask_prob).to(device)

    # create inputs
    inputs = batch.detach() * torch.logical_not(masks).to(device)
    inputs[inputs == 0] = 103

    # create labels
    labels = batch.detach() * masks

    return inputs.long(), labels.long()


def read_file(path):
    with open(path, 'r') as f:
        content = "".join(f.readlines())
    return content


def read_data(folderpath, max_files):
    files = os.listdir(folderpath)

    return [read_file(os.path.join(folderpath, file)) for file in files[:max_files]]


def tokenize(texts):
    """
    Parameters:
        texts (List[string]): A list of strings, each string represents a book etc.

    Returns:
        output (List[List[int]]): A list of list of ints, each list of ints represent a tokenized book
    """
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)) for text in texts]


def partition(ids, max_len):
    """
    partition id in ids into blocks of max_len,
    remove last block to make sure every block is the same size
    """

    return [torch.tensor([id[i:i+max_len] for i in range(0, len(id), max_len)][:-1], dtype=torch.int32)
            for id in ids]


def create_pg19_data(path, max_len, max_files):
    """
    :return: List[Tensor(length, max_len)], None
    """

    data = partition(tokenize(read_data(path, max_files=max_files)), max_len=max_len)

    return data


if __name__ == "__main__":
    x = tokenize(["hello world, what is going on", "hello wrold, wha  dsa dkwkoaksdm"])
    print(x)

