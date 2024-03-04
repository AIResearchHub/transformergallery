

from transformers import BertTokenizerFast, BartTokenizerFast

import logging
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)

import torch
import os
# from torch.nn.utils.rnn import CharTokenizer
from torchtext.data import get_tokenizer


def apply_mlm_mask(batch, mask_prob):
    """
    A function to apply masked language modeling for BERT.

    Parameters:
        batch (Tensor): The tensor with ids to be masked
        mask_prob (int): Masking probabilities for each token
    """
    device = batch.device

    probs = torch.rand(*batch.shape)
    masks = (probs < mask_prob).to(device)

    # create inputs
    inputs = batch.detach() * torch.logical_not(masks).to(device)
    inputs[inputs == 0] = 103

    # create labels
    labels = batch.detach() * masks

    return inputs.long(), labels.long()


def tokenize(texts, type):
    """
    Args:
         texts (List):
         type (string):
    """
    if type == "bert":
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # if type == "char":
    #     tokenizer = CharTokenizer()
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


def filter_empty(data, min_len=1):
    return [x for x in data if x.size(0) >= min_len]

