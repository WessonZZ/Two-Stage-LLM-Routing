#!/bin/bash

path='datasets/RouterBench/routerbench_5shot-perf-cost.csv'

python clustering.py \
    --path "${path}"\
    --n_clusters 4\
    --cost_weight 1.2\
    --if_scaler True\
    --group 1 #group 1 for embedllm