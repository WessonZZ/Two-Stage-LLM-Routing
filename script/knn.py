# -*- coding: utf-8 -*-
# @Time    : 2025/4/12 16:05
# @Author  : Wesson
# @FileName: knn

import json
import os
from tqdm import tqdm
import time
import ast
from collections import Counter

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DebertaV2Model, DebertaV2Tokenizer

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return self.texts[idx]

def tokenize(data, tokenizer, model, device, batch_size=64, max_length=512):

    all_hidden_states = []
    sample_prompts = data['prompt'].tolist()
    
    dataset = SimpleDataset(sample_prompts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for batch in tqdm(dataloader, desc="Processing batches"):
            # Tokenize the batch
            inputs = tokenizer(
                batch, 
                max_length=max_length, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt', 
                add_special_tokens=True
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Get model outputs
            outputs = model(**inputs)
            # Extract the [CLS] token embeddings (first token of each sequence)
            hidden_states = outputs['last_hidden_state'][:, 0, :].cpu()
            
            # Convert to list and add to results
            batch_hidden_states = hidden_states.tolist()
            all_hidden_states.extend(batch_hidden_states)
    
    return all_hidden_states


def load_model(model_path, device):
    # "microsoft/mdeberta-v3-base"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_path, truncation_side = 'left', padding = True)
    model = DebertaV2Model.from_pretrained(model_path)
    model = model.to(device)
    for name, para in model.named_parameters():
        para.requires_grad_(False)
    return tokenizer, model

def load_data(path):
    
    if path.endswith('csv'):
        data = pd.read_csv(path)
    elif path.endswith('json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = pd.DataFrame(json.load(f))
    else:
        pass

    return data


def compute_simi(data1, data2):
    return np.dot(data1, data2) / (np.linalg.norm(data1) * np.linalg.norm(data2))


def train(k = 200):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer, model = load_model("/root/mDeberta-v3-base", device)
    train_data = pd.read_csv('datasets/embedLLM/embed-val.csv')
    hiddens_train = tokenize(train_data, tokenizer, model, device)
    test_data = pd.read_csv('datasets/embedLLM/embed-test.csv')
    hiddens_test = tokenize(test_data, tokenizer, model, device)
    price = pd.read_csv('datasets/embedLLM/embedLLM-perf-cost.csv')
    selected_models = load_data('datasets/select_models_datasets.json')['embedllm1']['selected_models']
    selected_datas = load_data('datasets/select_models_datasets.json')['embedllm1']['selected_datasets']
    # selected_models = load_data('datasets/select_models_datasets.json')['routerbench']['selected_models']
    # selected_datas = load_data('datasets/select_models_datasets.json')['routerbench']['selected_datasets']
    scores = []
    costs = []
    score_ = 0
    cost_ = 0
    # print(test_data['dataset'])
    for i, row in tqdm(enumerate(hiddens_test), total = len(hiddens_test), desc = 'Sample'):
        # if test_data.loc[i, 'dataset'] in selected_datas:
        simi = [compute_simi(np.array(row), np.array(hiddens_train[j])) for j, d in enumerate(hiddens_train)]
        argmax_id = np.argsort(simi)[-k:]
        models = []
        for id in argmax_id:
            pos_models = ast.literal_eval(train_data.loc[id, 'pos'])
            neg_models = ast.literal_eval(train_data.loc[id, 'neg'])
            pos_models = pos_models + neg_models
            pos_models = [m for m in pos_models if m in selected_models]
            if len(pos_models) > 0:
                models.append(pos_models[0])
        argmax_model = Counter(models).most_common(1)[0][0]
        score_1 = test_data.loc[i, argmax_model]
        # cost_1 = test_data.loc[i, f'{argmax_model}|total_cost']
        cost_1 = price.loc[price['models'] == argmax_model, 'avg_cost'].values[0]
        scores.append(score_1.astype(int))
        costs.append(cost_1)
        score_ += score_1
        cost_ += cost_1
    
    print(f"Final score: {round(score_/len(scores),2)}, cost: {cost_}")
    with open('output/embedllm/knn-result.json', 'w', encoding='utf-8') as f:
        json.dump({'score': [score.item() if isinstance(score, np.generic) else score for score in scores],  # 转换为原生类型
                'cost': [cost.item() if isinstance(cost, np.generic) else cost for cost in costs]}, 
                f, indent=4)
        
train(k = 200)
    
    
    
    




