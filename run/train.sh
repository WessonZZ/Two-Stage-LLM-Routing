#!/bin/bash

MODEL="/root/mDeberta-v3-base"

LR=5e-4

python train_router.py \
    --model_path "${MODEL}"\
    --data_path 'datasets/embedLLM/embed-train.csv'\
    --eval_path 'datasets/embedLLM/embed-test.csv'\
    --embedding_path 'none'\
    --cluster_path 'datasets/embedLLM/cluster_7_cost_1.2-1.csv'\
    --max_seq_length 512\
    --dim 768\
    --is_ood False\
    --is_newmodels False\
    --simi 'cos'\
    --tau 0.4\
    --margin 0.3\
    --alpha 0.5\
    --beta 0.001\
    --batch_size 16\
    --eval_size 100000\
    --epochs 2\
    --warmup_rate 0.1\
    --lr $LR\
    --gradient_accumulation 4\
    --use_scheduler True\
    --scheduler 'cosine'\
    --c_steps 100\
    --eval_steps 50\
    --seed 42\
    --log_steps 5\
    --save_steps 100\
    --save_folder 'output/embedllm/cluster7'\
    --checkpoint_dir 'ck_dir'

