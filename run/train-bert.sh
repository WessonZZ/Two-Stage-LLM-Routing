#!/bin/bash

MODEL="/root/autodl-tmp/qwen2.5-0.5B"

LR=5e-4

python bert.py \
    --model_path "${MODEL}"\
    --data_path 'datasets/RouterBench/routerbench_5shot-train.csv'\
    --eval_path 'datasets/RouterBench/routerbench_5shot-test.csv'\
    --cluster_path 'datasets/RouterBench/cluster_2_cost_1.2.csv'\
    --max_seq_length 512\
    --dim 768\
    --n_models 9\
    --is_ood False\
    --is_newmodels False\
    --batch_size 16\
    --eval_size 100000\
    --epochs 2\
    --warmup_rate 0.1\
    --lr $LR\
    --gradient_accumulation 4\
    --use_scheduler True\
    --scheduler 'cosine'\
    --eval_steps 50\
    --seed 42\
    --log_steps 5\
    --save_steps 100\
    --save_folder 'output/routerbench/qwen2.5'\
    --checkpoint_dir 'ck_dir'