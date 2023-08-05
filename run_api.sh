#!/bin/bash

python3 -m torch.distributed.run \
    --nproc_per_node 2 api.py \
    --ckpt_dir /path/to/model/dir/ \
    --tokenizer_path /path/to/tokenizer.model \
    --max_seq_len 1024 \
    --max_batch_size 4