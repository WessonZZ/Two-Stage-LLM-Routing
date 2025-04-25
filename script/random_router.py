# -*- coding: utf-8 -*-
# @Time    : 2025/4/13 10:41
# @Author  : Wesson
# @FileName: random
import numpy as np
import pandas as pd
import json
import random

def load_data(path):
    
    if path.endswith('csv'):
        data = pd.read_csv(path)
    elif path.endswith('json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = pd.DataFrame(json.load(f))
    else:
        pass

    return data

test_data = load_data('datasets/embedLLM/embed-test.csv')
selected_models = load_data('datasets/select_models_datasets.json')['embedllm1']['selected_models']
selected_datas = load_data('datasets/select_models_datasets.json')['embedllm1']['selected_datasets']
price = pd.read_csv('datasets/embedLLM/embedLLM-perf-cost.csv')
scores = []
costs = []
score_ = 0
cost_ = 0

for i in range(len(test_data)):
    # if test_data.loc[i, 'dataset'] in selected_datas:
    selected_model = random.sample(selected_models, k=1)[0]
    score_ += test_data.loc[i, selected_model].astype(float)
    cost_ += price.loc[price['models'] == selected_model, 'avg_cost'].astype(float)
    # cost_ += test_data.loc[i, f"{selected_model}|total_cost"].astype(float)
    
    # 追加到 scores 和 costs 列表
    scores.append(test_data.loc[i, selected_model].astype(float))
    costs.append(price.loc[price['models'] == selected_model, 'avg_cost'].astype(float))
    # costs.append(test_data.loc[i, f"{selected_model}|total_cost"].astype(float))

# 输出最终的 score 和 cost
print(f"Final score: {round(score_/len(scores),2)}, cost: {cost_}")

# 序列化并保存为 JSON
with open('output/embedllm/random-result.json', 'w', encoding='utf-8') as f:
    json.dump({
        'score': [
            score.item() if isinstance(score, np.generic) else float(score) 
            for score in scores
        ],  # 转换为原生类型
        'cost': [
            cost.item() if isinstance(cost, np.generic) else float(cost) 
            for cost in costs
        ]
    }, f, indent=4)

