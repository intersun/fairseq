#!/usr/bin/env bash

MODEL_ARCH="$1"
PEAK_LR=$2        # 0.0005          # Peak learning rate, adjust as needed
UPDATE_FREQ=$3      # 16          # Increase the batch size 16x

DATA_DIR=/ssd2/roberta
OUTPUT_DIR=/ssd2/roberta_output/
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=32        # Number of sequences per batch (batch size)


mkdir -p $OUTPUT_DIR/$MODEL_ARCH/$MAX_SENTENCES-$UPDATE_FREQ-$PEAK_LR
python train.py --fp16 $DATA_DIR \
    --save-dir $OUTPUT_DIR/$MODEL_ARCH/$MAX_SENTENCES-$UPDATE_FREQ-$PEAK_LR \
    --task masked_lm --criterion masked_lm \
    --arch $MODEL_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
    --mask-whole-words --bpe gpt2  \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 2000 | tee log-$MODEL_ARCH-$MAX_SENTENCES-$UPDATE_FREQ-$PEAK_LR

# python -u train.py --fp16 /ssd2/roberta_small \
#  --save-dir /ssd2/init_run_small \
#  --total-num-update 125000 \
#  --max-update 125000 \
#  --warmup-updates 1000 \
#  --lr 4e-3 \
#  --max-sentences 32 \
#  --update-freq 4 \
#  --task masked_lm --criterion masked_lm \
#  --arch roberta_base --sample-break-mode complete --tokens-per-sample 512 \
#  --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
#  --lr-scheduler polynomial_decay \
#  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
#  --log-format simple --log-interval 1 \
#  --mask-whole-words --bpe gpt2  \
#  --skip-invalid-size-inputs-valid-test \
#  --save-interval-updates 1000 > /ssd2/init_run_small/log
#
