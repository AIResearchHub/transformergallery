from pytorch_pretrained_bert import BertTokenizer

import logging
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)

import torch
import os
from torch.nn.utils.rnn import CharTokenizer
from torchtext.data import get_tokenizer

def read_file(path):
    with open(path, 'r') as f:
        content = "".join(f.readlines())
    return content


def read_data(folderpath, max_files):
    files = os.listdir(folderpath)

    return [read_file(os.path.join(folderpath, file)) for file in files[:max_files]]


def tokenize(texts,type):
    if type == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if type == "char":
        tokenizer = CharTokenizer()
    if type == "spacy":
        tokenizer = get_tokenizer("spacy")


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
