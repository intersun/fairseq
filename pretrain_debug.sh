DEBUG=false
if ! $DEBUG; then
    echo "normal training"
    pip install --user --editable .
    fairseq-train --fp16 /ssd2/roberta \
      --save-dir /ssd2/init_run \
      --total-num-update 12500 \
      --max-update 12500 \
      --warmup-updates 1000 \
      --lr 4e-3 \
      --max-sentences 32 \
      --update-freq 4 \
      --task masked_lm --criterion masked_lm \
      --arch roberta_base --sample-break-mode complete --tokens-per-sample 512 \
      --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
      --lr-scheduler polynomial_decay \
      --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
      --log-format simple --log-interval 1 \
      --mask-whole-words

    if false; then
        TOTAL_UPDATES=125000    # Total number of training steps
        WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
        PEAK_LR=0.0005          # Peak learning rate, adjust as needed
        TOKENS_PER_SAMPLE=512   # Max sequence length
        MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
        MAX_SENTENCES=16        # Number of sequences per batch (batch size)
        UPDATE_FREQ=16          # Increase the batch size 16x

        DATA_DIR=data-bin/wikitext-103

        fairseq-train --fp16 $DATA_DIR \
            --task masked_lm --criterion masked_lm \
            --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
            --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
            --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
            --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
            --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
            --max-update $TOTAL_UPDATES --log-format simple --log-interval 1
    fi
else

    echo "debug training"
    TOTAL_UPDATES=32  # Total number of training steps
    WARMUP_UPDATES=10    # Warmup the learning rate over this many updates
    PEAK_LR=0.0005          # Peak learning rate, adjust as needed
    TOKENS_PER_SAMPLE=512   # Max sequence length
    MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
    MAX_SENTENCES=32        # Number of sequences per batch (batch size)
    UPDATE_FREQ=2          # Increase the batch size 16x

    DATA_DIR=/datadrive_b/roberta/roberta_data/binary_debug

    fairseq-train --fp16 $DATA_DIR \
        --task masked_lm --criterion masked_lm \
        --arch roberta_medium --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
        --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
        --save-dir /datadrive_b/roberta/output/checkpoints \
        --mask-whole-words
fi
