from collections import Counter
from itertools import zip_longest
import logging
from multiprocessing import Pool
import os
import shutil
import sys
import torch
import random

from fairseq import (
    distributed_utils,
    options,
)

from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer

from fairseq_cli.preprocess import main as main_preprocess
from fairseq_cli.train import main as main_train
from fairseq_cli.train import distributed_main as distributed_main_train


#################################################################################################################
# pre-process below
#################################################################################################################

if False:
      cmd = '--only-source --srcdict bpe/gpt2/dict.txt ' \
            '--trainpref /datadrive_b/roberta/wikitext-103-raw/wiki.valid.bpe ' \
            '--validpref /datadrive_b/roberta/wikitext-103-raw/wiki.valid.bpe ' \
            '--testpref /datadrive_b/roberta/wikitext-103-raw/wiki.test.bpe ' \
            '--destdir data-bin/wikitext-103 ' \
            '--workers 1 '

      cmd = cmd.split()

      parser = options.get_preprocessing_parser()
      args = parser.parse_args(cmd)
      main_preprocess(args)



#################################################################################################################
# train below
#################################################################################################################

TOTAL_UPDATES = 10   # Total number of training steps
WARMUP_UPDATES = 2   # Warmup the learning rate over this many updates
PEAK_LR = 0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE = 512   # Max sequence length
MAX_POSITIONS = 512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES = 16        # Number of sequences per batch (batch size)
UPDATE_FREQ = 16          # Increase the batch size 16x

# DATA_DIR = 'data-bin/wikitext-103'

DATA_DIR = '/datadrive_b/roberta/roberta_data/binary_debug'

cmd = f'{DATA_DIR} ' \
      f'--task masked_lm --criterion masked_lm '\
      f'--arch roberta_base --sample-break-mode complete --tokens-per-sample {TOKENS_PER_SAMPLE} '\
      f'--optimizer adam --adam-betas \'(0.9,0.98)\' --adam-eps 1e-6 --clip-norm 0.0 ' \
      f'--lr-scheduler polynomial_decay --lr {PEAK_LR} --warmup-updates {WARMUP_UPDATES} ' \
      f'--total-num-update {TOTAL_UPDATES} '\
      f'--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 '\
      f'--max-sentences {MAX_SENTENCES} --update-freq {UPDATE_FREQ} '\
      f'--max-update {TOTAL_UPDATES} --log-format simple --log-interval 1 --fp16 ' \
      f'--mask-whole-words --bpe gpt2 '

parser = options.get_training_parser()
args = options.parse_args_and_arch(parser, input_args=cmd.split(), modify_parser=None)

if args.distributed_init_method is None:
      distributed_utils.infer_init_method(args)

# single GPU training
main_train(args)