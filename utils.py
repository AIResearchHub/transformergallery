

from transformers import BertTokenizerFast, BartTokenizerFast

import logging
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F
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


def tokenize(texts, type='bert'):
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


def remove_padding(data, sep_token=102, start_token=50, end_token=51):
    """
    Args:
        data (tensor): data to remove padding
        sep_token (int): Id associated with [SEP]
        start_token (int): Id associated with [START]
        end_token (int): Id associated with [END]

    Returns:
        data (tensor): data after removing padding
        mask (tensor): mask of padding
    """
    mask = (data != sep_token) & (data != start_token) & (data != end_token)

    return data[mask], mask


@torch.inference_mode()
def generate_samples(model, prompt_ids=[50, 102], seq_len=512,
                     B=8, T=500, temperature=0.5):
    """
    Note:
        [START] = 50
        [SEP] = 102
        [END] = 51

    Args:
        model (nn.Module): trained model to generate samples with
        prompt_ids (List[int]): A list of ids of the starting prompt
        B (int): batch size
        T (int): generate timesteps
        temperature (float): Controls the "creativity" of the text generated always between 0 and 1
                             higher (e.g. 0.7) results in more diverse and creative outputs
                             lower (e.g. 0.2) makes the output more deterministic and focused
        device (string): which device to run on, cpu or cuda

    Returns:
        x (tensor): Generated samples in tensor
    """
    if len(prompt_ids) == 0:
        prompt_ids = [50, 102]

    # (B, T, vocab_size)
    x = torch.tensor(prompt_ids, dtype=torch.int64).repeat(B, 1)
    length = len(prompt_ids)

    # reset recurrent state and cache recurrent state
    model.module.reset()
    states = model.module.get_state()
    for _ in range(T):
        model.module.set_state(states)

        start = ((length-1) // seq_len) * seq_len
        logits = model(x[:, start:].cuda())
        logits = logits.cpu()

        # only update recurrent state if length has just passed seq_len
        # to utilize recurrent state for the next seq_len
        if length % seq_len == 0:
           states = model.module.get_state()

        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        x = torch.concat((x, next_id), dim=-1)
        length += 1

    return x.cpu()

