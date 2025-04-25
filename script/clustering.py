# -*- coding: utf-8 -*-
# @Time    : 2025/4/7 20:39
# @Author  : Wesson
# @FileName: clustering

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from scipy.special import softmax

import json
from argparse import ArgumentParser, ArgumentTypeError

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Chalkboard SE'
plt.rcParams['font.size'] = 10

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler


def str_to_bool(value):
    if value.lower() in ("true", "yes", "t", "y", "1"):
        return True
    elif value.lower() in ("false", "no", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def parse_config():
    parser = ArgumentParser()
    parser.add_argument("--path",            type = str)
    parser.add_argument("--n_clusters",      type = int, default = 4)
    parser.add_argument("--cost_weight",     type = float, default = 2.0)
    parser.add_argument("--if_scaler",       type = str_to_bool, default=False)
    parser.add_argument("--group",           type = int, default = 1)
    
    return parser.parse_args()
    

def load_data(path):
    
    if path.endswith('csv'):
        return pd.read_csv(path)
    elif path.endswith('json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        pass


def kmeans(n_clusters, path, selected_models_datasets, cost_weight = 2, if_scaler = True, group = 1):
    
    root_path = "/".join(path.split('/')[:-1])
    data = load_data(path)
    
    if 'routerbench' in path.lower():
        new_models = selected_models_datasets['routerbench']['new_models']
        new_datasets = selected_models_datasets['routerbench']['new_datasets']
        columns = [c for c in data.columns.tolist() if c not in new_datasets] #OOD datasets
        data = data.loc[:, columns]
        data_ = data.loc[~data['models'].isin(new_models), :]
        
        save_path = f'{root_path}/cluster_{n_clusters}_cost_{cost_weight}.csv'
        
    elif 'embedllm' in path.lower():
        selected_models = selected_models_datasets[f'embedllm{group}']['selected_models']
        new_models = selected_models_datasets[f'embedllm{group}']['new_models']
        new_datasets = selected_models_datasets[f'embedllm{group}']['new_datasets']
        columns = [c for c in data.columns.tolist() if c not in new_datasets] #OOD datasets
        data = data.loc[:, columns]
        data_ = data.loc[data['models'].isin(selected_models), :]
        
        save_path = f'{root_path}/cluster_{n_clusters}_cost_{cost_weight}-{group}.csv'

    # columns = [c for c in columns if c not in ['avg_cost']]
    # print(data)
    data_['avg_perf'] = data_.loc[:, columns].drop(columns= ['models', 'avg_cost']).mean(axis = 1)
    data_['std_perf'] = data_.loc[:, columns].drop(columns= ['models', 'avg_cost']).std(axis = 1)
    # print(data_)
    
    X = data_.drop(columns = 'models')
    cost_scaled = X[['avg_cost']]
    if if_scaler:
        scaler = MinMaxScaler()
        cost_scaled = scaler.fit_transform(cost_scaled)
    X['avg_cost_norm'] = cost_scaled * cost_weight
    # print(X)
    
    print(f"Starting kmeans clustering, clusters = {n_clusters} ...\n")
    kmeans = KMeans(n_clusters= n_clusters, init = 'random', n_init = 100, random_state = 999)
    kmeans.fit(X.drop(columns='avg_cost'))
    
    models = data_['models'].tolist()
    clusters = list(kmeans.labels_)
    clusters_pd = pd.DataFrame({'models': models, 'cluster': clusters})
    clusters_pd = clusters_pd.merge(data_, how = 'left', on = 'models')
    # clusters_pd['avg_cost_norm'] = X['avg_cost_norm']
    cluster_cost_perf = clusters_pd.groupby('cluster')[['avg_perf', 'avg_cost']].mean().reset_index().rename(columns = {'avg_perf': 'cluster_perf', 'avg_cost': 'cluster_cost'})
    # print(clusters_pd)
    new_data = data.loc[data['models'].isin(new_models), :]
    # print(data)
    new_data['avg_perf'] = new_data.loc[:, columns].drop(columns= ['models', 'avg_cost']).mean(axis = 1)
    new_data['std_perf'] = new_data.loc[:, columns].drop(columns= ['models', 'avg_cost']).std(axis = 1)
    new_data['avg_cost_norm'] = cost_weight * scaler.transform(new_data[['avg_cost']])
    # print(new_data)
    new_clusters_pd = pd.DataFrame({'models': list(new_data['models']), 'cluster': list(kmeans.predict(new_data.drop(columns = ['models', 'avg_cost'])))})
    new_clusters_pd = new_clusters_pd.merge(new_data.drop(columns='avg_cost_norm'), how = 'left', on = 'models')
    clusters_pd = pd.concat([clusters_pd, new_clusters_pd], axis = 0)
    clusters_pd = clusters_pd.merge(cluster_cost_perf, how='left', on = 'cluster')
    clusters_pd['act_cluster'] = clusters_pd['cluster_cost'].rank(method = 'dense', ascending= True).astype(int) - 1
    
    clusters_pd.to_csv(save_path, index = False)
    print(f"Clustering finished.\n")
    print(clusters_pd.loc[:, ['models', 'avg_perf', 'avg_cost', 'cluster_perf', 'cluster_cost', 'cluster', 'act_cluster']])
    plot_cluster(clusters_pd, n_clusters, cost_weight, path, group)
    

def plot_cluster(data, n_clusters, cost_weight, path, group = 2):
    ss = ['o', 's', '^', 'D', '*', '+', 'x', 'v', '>', '<', '|']
    models = data['models'].tolist()
    cost_all = data['avg_cost'].tolist()
    perf_all = data['avg_perf'].tolist()
    clusters = data['cluster'].tolist()
    root_path = "/".join(path.split('/')[:-1])
    
    if 'routerbench' in path.lower():
        save_path = f'{root_path}/figure-cluster_{n_clusters}_cost_{cost_weight}.pdf'
    elif 'embedllm' in path.lower():
        save_path = f'{root_path}/figure-cluster_{n_clusters}_cost_{cost_weight}-{group}.pdf'
    
    fig, ax = plt.subplots(1,1, figsize = (7, 6))
    for i, model in enumerate(models):
        ax.scatter(cost_all[i], perf_all[i], label = model, marker=ss[clusters[i]])
        ax.grid()
        ax.text(cost_all[i] - 0.05, perf_all[i] + 0.005, s = model.split('/')[-1] if '/' in model else model)

    plt.ylabel('Avg Accuracy', fontdict={'size': 14})
    plt.xlabel('Avg Cost/$', fontdict={'size': 14})
    # plt.tight_layout()
    plt.savefig(save_path, dpi = 300)
    plt.show()



if __name__ == "__main__":
    
    config = parse_config()
    selected_models_datasets = load_data(path = 'datasets/select_models_datasets.json')
    kmeans(config.n_clusters, config.path, selected_models_datasets, config.cost_weight, config.if_scaler, config.group)

