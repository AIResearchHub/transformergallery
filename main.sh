#!/bin/bash

SEQ_LEN=512
VOCAB_SIZE=30522
N_LAYERS=4
D_MODEL=768
N_HEAD=8
P=0.1
LR=0.00004
BATCH_SIZE=32
BURNIN=0
ROLLOUT=5
WARMUP_STEPS=100
DEVICE="cuda:0"
CACHE_DIR="./cache/datasets"

python main.py \
--seq_len $SEQ_LEN \
--vocab_size $VOCAB_SIZE \
--n_layers $N_LAYERS \
--d_model $D_MODEL \
--n_head $N_HEAD \
--p $P \
--lr $LR \
--batch_size $BATCH_SIZE \
--burnin $BURNIN \
--rollout $ROLLOUT \
--warmup_steps $WARMUP_STEPS \
--device $DEVICE \
--cache_dir $CACHE_DIR
