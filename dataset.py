from pytorch_pretrained_bert import BertTokenizer

import logging
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)

import torch
import os


def read_data(folderpath, max_files=500):
    files = os.listdir(folderpath)

    data = []
    for file in files[:max_files]:

        path = os.path.join(folderpath, file)
        with open(path, 'r') as f:
            content = "".join(f.readlines())
            data.append(content)

    return data


def tokenize(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ids = []
    for text in texts:
        ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))

    return ids


def partition(ids, max_len):
    """
    partition id in ids into blocks of max_len,
    remove last block to make sure every block is the same size
    """

    books = []
    for id in ids:
        book = torch.tensor([id[i:i+max_len] for i in range(0, len(id), max_len)][:-1], dtype=torch.int32)
        if book.size(0) > 30:
            books.append(book)

    for book in books:
        print(book.shape)

    return books


def create_bert_data(path="data/pg19/train", max_len=512, total_len=30, max_files=30):
    """
    :return: List[Tensor(length, max_len)], None
    """

    data = partition(tokenize(read_data(path, max_files=max_files)),
                     max_len=max_len,
                     total_len=total_len
                     )

    return data

