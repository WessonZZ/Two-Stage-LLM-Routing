# -*- coding: utf-8 -*-
# @Time    : 2025/4/8 17:48
# @Author  : Wesson
# @FileName: Dataclass

import os
import json
import ast

import pandas as pd
import numpy as np
from scipy.special import softmax
from collections import Counter

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class RouterData(Dataset):
    def __init__(self, data_path, cluster_path, max_seq_length, tokenizer, is_ood, is_newmodels):
        super().__init__()
        self.is_ood = is_ood
        self.is_newmodels = is_newmodels
        self.data_path = data_path
        self.cluster_path = cluster_path
        self.data_name = 'embedllm' if 'embedllm' in self.data_path.lower() else 'routerbench'
        self.group = self.cluster_path.split('.csv')[0][-1]  if self.data_name == 'embedllm' else 1 # for group1 or group2 of embedLLM
        self.data = self.select_model_dataset()

        self.tokenizer = tokenizer
        # self.tokenizer.truncation_side = "left"
        self.max_seq_length = max_seq_length

    def load_data(self, path):
        
        if path.endswith('csv'):
            data = pd.read_csv(path)
        elif path.endswith('json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            pass

        return data

    def select_model_dataset(self):
        data = self.load_data(self.data_path).fillna(0)
        selected = self.load_data('datasets/select_models_datasets.json')
        
        if self.data_name == 'routerbench':
            self.selected_models = selected['routerbench']['selected_models']
            self.new_models = selected['routerbench']['new_models'] #selected as OOD models
            self.new_datasets = selected['routerbench']['new_datasets'] #selected as OOD datasets
            self.select_datasets = selected['routerbench']['selected_datasets'] 
            self._load_clusters()
            self.columns = [c for c in data.columns.tolist() if c not in self.new_models]
        elif self.data_name == 'embedllm':
            self.selected_models = selected[f'embedllm{self.group}']['selected_models']
            self.new_models = selected[f'embedllm{self.group}']['new_models'] #selected as OOD models
            self.new_datasets = selected[f'embedllm{self.group}']['new_datasets'] #selected as OOD datasets
            self.select_datasets = selected[f'embedllm{self.group}']['selected_datasets'] 
            self._load_clusters()
            self.columns = [c for c in data.columns.tolist() if c in self.selected_models] + ['prompt_id', 'category', 'prompt', 'dataset', 'pos', 'neg']
        else:
            pass
        if self.is_ood:
            self.new_datasets = []
        data = data.loc[~data['dataset'].isin(self.new_datasets), self.columns]
        # data = data[data['pos'].apply(lambda x: isinstance(x, str) and len(ast.literal_eval(x)) > 0 if x else False)]
        # data = data[data['neg'].apply(lambda x: isinstance(x, str) and len(ast.literal_eval(x)) > 0 if x else False)]
        
        return data

    
    def _load_clusters(self):
        
        # scaler = MinMaxScaler()
        self.perf = self.load_data(self.cluster_path)
        self.all_model_clusters = dict(zip(self.perf['models'], self.perf['act_cluster']))
        self.clusters = self.perf['act_cluster'].tolist()
        self.n_clusters = max(self.clusters) + 1
        
        if self.is_newmodels:
            self.selected_models = self.selected_models + self.new_models
            self.new_models = []
        self.model_clusters = {m: c for m, c in self.all_model_clusters.items() if str(m) in self.selected_models}
        self.new_model_clusters = {m: c for m, c in self.all_model_clusters.items() if str(m) in self.new_models}
        self.n_models = len(self.selected_models)
        self.model_orders = {m: i for i, m in enumerate(self.selected_models + self.new_models)}
        self.perf_dimensions = self.select_datasets + ['avg_perf', 'std_perf', 'avg_cost']
        # print("Dataclass", self.model_orders)


    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, index):
        
        data_point = self.data.iloc[index].to_dict()
        prompt = data_point['prompt']
        pos_models = [m for m in ast.literal_eval(data_point['pos']) if m in self.selected_models]
        neg_models = [m for m in ast.literal_eval(data_point['neg']) if m in self.selected_models]
        model_labels = {m: data_point[m] for m in self.selected_models}
        if self.data_name == 'routerbench':
            model_costs = {self.model_orders[m]: data_point[f'{m}|total_cost'] for m in pos_models + neg_models}
            # print(model_costs)
        elif self.data_name == 'embedllm':
            model_costs = {}
            for m in pos_models + neg_models:
                model_costs[self.model_orders[m]] = self.perf.loc[self.perf['models'] == m, 'avg_cost'].values[0]
        # print(data_point)
        model_costs = torch.stack([torch.tensor(v) for _, v in model_costs.items()])
        # model_clusters = {self.model_orders[m]: self.model_clusters[m] for m in self.selected_models}
        # model_clusters = torch.stack([torch.tensor(v) for _, v in sorted(model_clusters.items())])
        return self.tokenize(prompt, pos_models, neg_models, model_labels, model_costs)
    
    def get_most_common_clusters(self, pos, neg):
        # 统计每个模型所属簇的频次
        pos_1 = pos
        most_common_clusters = []

        counts = Counter([self.model_clusters[m] for m in pos_1])
        # 获取出现次数最多的簇，按频次降序排列
        most_common_clusters = [k for k, v in counts.most_common(1)]
        
        # 如果只找到了一个簇，继续扩展pos_1，加入一个负样本
        if len(most_common_clusters) < 1:
            needed_neg_samples = 1 - len(most_common_clusters)
            # 确保不会超出负样本的范围
            pos_1 += neg[:needed_neg_samples]
            # 重新计算频次
            counts = Counter([self.model_clusters[m] for m in pos_1])
            # 获取新的簇
            most_common_clusters = [k for k, v in counts.most_common(1)]
        
        # 返回出现最多的簇ID
        return most_common_clusters

    
    def tokenize(self, prompt, pos, neg, model_labels, model_costs):
        

        inputs = self.tokenizer(prompt, max_length=self.max_seq_length, padding='max_length', truncation = True, return_tensors='pt', add_special_tokens = True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # if len() > self.max_seq_length:
        #     tokens = tokens[-self.max_seq_length:]  # left truncation
        # inputs = self.tokenizer.convert_tokens_to_ids(tokens)
        # inputs = self.tokenizer.prepare_for_model(inputs, max_length=self.max_seq_length, padding='max_length', return_tensors='pt', add_special_tokens = True)

        cluster_pos = self.get_most_common_clusters(pos, neg) #the cluster who has the most positive models as the positive clusters
        # print(len(cluster_pos))
        cluster_neg = torch.tensor(list(set([c for c in self.clusters if c not in cluster_pos])))
        cluster_pos = torch.tensor(cluster_pos)
        
        model_index = torch.ones(self.n_models) * (-100)
        model_labels_ = torch.ones(self.n_models) * (-100)
        
        model_index[:len(pos)] = torch.tensor([self.model_orders[m] for m in pos])
        model_index[len(pos):] = torch.tensor([self.model_orders[m] for m in neg])
        model_labels_[:len(pos)] = torch.tensor([model_labels[m] for m in pos])
        model_labels_[len(pos):] = torch.tensor([model_labels[m] for m in neg])
        # model_costs = torch.stack([torch.tensor(v) for _, v in sorted(model_costs.items())])
        
        #for bert only
        # model_index_ = {m: i for i, m in enumerate(self.selected_models)}
        # model_index[:len(pos)] = torch.tensor([model_index_[m] for m in pos])
        # model_index[len(pos):] = torch.tensor([model_index_[m] for m in neg])
        
        return inputs,\
                cluster_pos,\
                cluster_neg,\
                model_index,\
                model_labels_,\
                model_costs,\
                len(pos)
        
            

if __name__ == "__main__":
    
    class config:
        data_path = 'datasets/embedLLM/embed-val.csv'
        cluster_path = 'datasets/embedLLM/cluster_6_cost_1.2-1.csv'
        embedding_path = "None"
        max_seq_length = 512
        dim = 768
        is_ood = True
        is_newmodels = False
        
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", truncation_side = 'left', padding = True)
        
    data = RouterData(config.data_path, config.cluster_path, config.embedding_path, config.max_seq_length, tokenizer, config.is_ood, config.is_newmodels)
    
    dataloader = DataLoader(data, batch_size= 4, shuffle = False)
    for batch in dataloader:
        print(batch)
        break