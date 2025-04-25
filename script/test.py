

import pandas as pd
import ast
from tqdm import tqdm
import json


def load_data(path):
    
    if path.endswith('csv'):
        data = pd.read_csv(path)
    elif path.endswith('json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        pass

    return data

def all():
    data = pd.read_csv('datasets/RouterBench/routerbench_5shot-perf-cost.csv')
    Costs = dict(zip(data['models'], data['avg_cost']))
    print(data.drop(columns = 'models').mean(axis = 0))
    print(data.drop(columns = 'models').mean(axis = 0).mean())
    models = data['models'].tolist()
    selected_datasets = load_data('datasets/select_models_datasets.json')['routerbench']['selected_datasets']
    selected_models = load_data('datasets/select_models_datasets.json')['routerbench']['selected_models']
    data_test = pd.read_csv('datasets/RouterBench/routerbench_5shot-test.csv')
    data_test = data_test.fillna(0)

    data_test = data_test.fillna(0)
    score = 0
    cost = 0
    score_ind = {}
    cost_ind = {}
    j = 0
    for i in tqdm(range(len(data_test)), total=len(data_test), desc='Sample'):
        data_point = data_test.iloc[i,:].to_dict()
        if data_point['dataset'] in selected_datasets:
            pos = ast.literal_eval(data_point['pos'])
            neg = ast.literal_eval(data_point['neg'])
            for m_ in pos + neg:
                if m_ in selected_models:
                    break
            model_ = m_
            # print(model_)
            score_ = data_point[model_]
            # cost_ = data_point[model_ + '|total_cost']
            cost_ = Costs[model_]
            score += score_
            cost += cost_
            j += 1
            for m in selected_models:
                if m not in score_ind:
                    score_ind[m] = 0
                    cost_ind[m] = 0
                if data_point[m] == None:
                    score_ind[m] += 0
                else:
                    score_ind[m] += data_point[m]
                    cost_ind[m] += data_point[m + '|total_cost']
                    # cost_ind[m] += Costs[m]

    print(score/j, cost)
    save_data = {}
    for m in selected_models:
        save_data[m] = [score_ind[m]/j, cost_ind[m]]
    save_data['oracle'] = [score/j, cost]

    with open('datasets/RouterBench/model_perf_cost-test-selected_datasets.json', 'w', encoding= 'utf-8') as f:
        json.dump(save_data, f, indent = 4)


def single():
    data = pd.read_csv('datasets/RouterBench/routerbench_5shot-perf-cost.csv')
    Costs = dict(zip(data['models'], data['avg_cost']))
    print(data.drop(columns = 'models').mean(axis = 0))
    print(data.drop(columns = 'models').mean(axis = 0).mean())
    models = data['models'].tolist()
    selected_datasets = load_data('datasets/select_models_datasets.json')['routerbench']['selected_datasets']
    selected_models = load_data('datasets/select_models_datasets.json')['routerbench']['selected_models']
    data_test = pd.read_csv('datasets/RouterBench/routerbench_5shot-test.csv')
    data_test = data_test.fillna(0)

    data_test = data_test.fillna(0)
    final_data = {}
    j = 0
    for i in tqdm(range(len(data_test)), total=len(data_test), desc='Sample'):
        data_point = data_test.iloc[i,:].to_dict()
        if data_point['dataset'] not in final_data:
            final_data[data_point['dataset']] = {}
        for m in selected_models:
            if m not in final_data[data_point['dataset']]:
                final_data[data_point['dataset']][m] = {'score': [], 'cost': []}
            final_data[data_point['dataset']][m]['score'].append(data_point[m])
            # cost_ = data_point[model_ + '|total_cost']
            final_data[data_point['dataset']][m]['cost'].append(data_point[m + '|total_cost'])
            # final_data[data_point['dataset']][m]['cost'].append(Costs[m])
        if 'oracle' not in final_data[data_point['dataset']]:
            final_data[data_point['dataset']]['oracle'] = {'score': [], 'cost': []}
        pos = ast.literal_eval(data_point['pos'])
        neg = ast.literal_eval(data_point['neg'])
        pos = pos + neg
        for mm in pos:
            if mm in selected_models:
                break
        mm_ = mm
        final_data[data_point['dataset']]['oracle']['score'].append(data_point[mm_])
        final_data[data_point['dataset']]['oracle']['cost'].append(Costs[mm_])


    with open('datasets/RouterBench/model_perf_cost-test-single_datasets.json', 'w', encoding= 'utf-8') as f:
        json.dump(final_data, f, indent = 4)
        

if __name__ == "__main__":
    single()
        

    
    