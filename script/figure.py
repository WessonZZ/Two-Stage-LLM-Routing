# -*- coding: utf-8 -*-
# @Time    : 2025/4/13 19:23
# @Author  : Wesson
# @FileName: figure

import pandas as pd
import numpy as np
from collections import Counter
import random
import ast
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Chalkboard SE'  # Mac 上的 Chalkboard 字体
plt.rcParams['font.size'] = 12 


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
    fig, ax = plt.subplots(2,2, figsize = (10, 8))
    
    selected_datasets = load_data('datasets/select_models_datasets.json')['embedllm1']['selected_datasets']
    selected_models = load_data('datasets/select_models_datasets.json')['embedllm1']['selected_models']
    
    test_data = load_data('datasets/embedLLM/embed-test.csv')
    bert_labels = load_data('output/embedllm/bert/bert-2025-04-13 17:20:51/evaluation-step-400-bert-False.json')
    bert_scores = bert_labels['scores']
    bert_costs = bert_labels['costs']
    bert_non_ood_score = [s for d, s in zip(test_data['dataset'].tolist(), bert_scores) if d in selected_datasets]
    bert_non_ood_cost = [c for d, c in zip(test_data['dataset'].tolist(), bert_costs) if d in selected_datasets]
    bert_non_ood_acc = round(sum(bert_non_ood_score)/len(bert_non_ood_score), 4)
    bert_ood_acc = round(sum(bert_scores)/len(bert_scores), 4)
    
    router_labels = load_data('output/embedllm/cluster2/2025-04-13 14:19:31/evaluation-step-150-newmodels-False.json')
    router_labels1 = load_data('output/embedllm/cluster3/2025-04-13 14:41:38/evaluation-step-500-newmodels-False.json')
    router_scores = router_labels1['scores']
    router_costs = router_labels['costs']
    router_non_ood_score = [s for d, s in zip(test_data['dataset'].tolist(), router_scores) if d in selected_datasets]
    router_non_ood_cost = [c for d, c in zip(test_data['dataset'].tolist(), router_costs) if d in selected_datasets]
    router_non_ood_acc = round(sum(router_non_ood_score)/len(router_non_ood_score), 4)
    router_ood_acc = round(sum(router_scores)/len(router_scores), 4)
    
    qwen_labels = load_data('output/embedllm/qwen2.5/bert-2025-04-13 17:37:49/evaluation-step-200-bert-False.json')
    qwen_scores = qwen_labels['scores']
    qwen_costs = qwen_labels['costs']
    qwen_non_ood_score = [s for d, s in zip(test_data['dataset'].tolist(), qwen_scores) if d in selected_datasets]
    qwen_non_ood_cost = [c for d, c in zip(test_data['dataset'].tolist(), qwen_costs) if d in selected_datasets]
    qwen_non_ood_acc = round(sum(qwen_non_ood_score)/len(qwen_non_ood_score), 4)
    qwen_ood_acc = round(sum(qwen_scores)/len(qwen_scores), 4)
    
    knn_labels = load_data('output/embedllm/knn-result.json')
    knn_scores = knn_labels['score']
    knn_costs = knn_labels['cost']
    knn_non_ood_score = [s for d, s in zip(test_data['dataset'].tolist(), knn_scores) if d in selected_datasets]
    knn_non_ood_cost = [c for d, c in zip(test_data['dataset'].tolist(), knn_costs) if d in selected_datasets]
    knn_non_ood_acc = round(sum(knn_non_ood_score)/len(knn_non_ood_score), 4)
    knn_ood_acc = round(sum(knn_scores)/len(knn_scores), 4)
    
    random_labels = load_data('output/embedllm/random-result.json')
    random_scores = random_labels['score']
    random_costs = random_labels['cost']
    random_non_ood_score = [s for d, s in zip(test_data['dataset'].tolist(), random_scores) if d in selected_datasets]
    random_non_ood_cost = [c for d, c in zip(test_data['dataset'].tolist(), random_costs) if d in selected_datasets]
    random_non_ood_acc = round(sum(random_non_ood_score)/len(random_non_ood_score), 4)
    random_ood_acc = round(sum(random_scores)/len(random_scores), 4)
    
    single_models_non_ood = load_data('datasets/embedLLM/model_perf_cost-test-select_datasets.json')
    single_models = load_data('datasets/embedLLM/model_perf_cost-test.json')
    
    print("Score: embedllm")
    print(f"Router non-ood: {router_non_ood_acc}, cost: {sum(router_non_ood_cost)}, ood: {router_ood_acc}, cost: {sum(router_costs)}")
    print(f"qwen non-ood: {qwen_non_ood_acc}, cost: {sum(qwen_non_ood_cost)}, ood: {qwen_ood_acc}, cost: {sum(qwen_costs)}")
    print(f"Bert non-ood: {bert_non_ood_acc}, cost: {sum(bert_non_ood_cost)}, ood: {bert_ood_acc}, cost: {sum(bert_costs)}")
    print(f"knn non-ood: {knn_non_ood_acc}, cost: {sum(knn_non_ood_cost)}, ood: {knn_ood_acc}, cost: {sum(knn_costs)}")
    print(f"random non-ood: {random_non_ood_acc}, cost: {sum(random_non_ood_cost)}, ood: {random_ood_acc}, cost: {sum(random_costs)}")
    for m in selected_models  + ['oracle']:
        print(f"{m}: ood: {single_models[m][0]}, cost: {single_models[m][1]}, non-ood: {single_models_non_ood[m][0]}, cost: {single_models_non_ood[m][1]}")
    print('\n')
    
    
    indices = np.arange(1, len(test_data) + 1)
    ax[0][0].plot(np.cumsum(router_costs), np.cumsum(router_scores)/indices, label = 'Router')
    ax[0][0].plot(np.cumsum(bert_costs), np.cumsum(bert_scores)/indices, label = 'Bert')
    ax[0][0].plot(np.cumsum(qwen_costs), np.cumsum(qwen_scores)/indices, label = 'Qwen2.5-0.5B')
    ax[0][0].plot(np.cumsum(knn_costs), np.cumsum(knn_scores)/indices, label = 'Knn')
    ax[0][0].plot(np.cumsum(random_costs), np.cumsum(random_scores)/indices, label = 'Random')
    ax[0][0].set_ylabel('Accuracy')
    ax[0][0].set_title('EmbedLLM all tasks')
    ax[0][0].set_ylim([0, 0.9])
    for m in selected_models:
        ax[0][0].scatter(single_models[m][1], single_models[m][0])
    ax[0][0].scatter(single_models['oracle'][1], single_models['oracle'][0], marker = '*', c = [223/255, 122/255, 94/255])
    
    indices = np.arange(1, len(random_non_ood_score) + 1)
    ax[0][1].plot(np.cumsum(router_non_ood_cost), np.cumsum(router_non_ood_score)/indices, label = 'Router')
    ax[0][1].plot(np.cumsum(bert_non_ood_cost), np.cumsum(bert_non_ood_score)/indices, label = 'Bert')
    ax[0][1].plot(np.cumsum(qwen_non_ood_cost), np.cumsum(qwen_non_ood_score)/indices, label = 'Qwen2.5-0.5B')
    ax[0][1].plot(np.cumsum(knn_non_ood_cost), np.cumsum(knn_non_ood_score)/indices, label = 'Knn')
    ax[0][1].plot(np.cumsum(random_non_ood_cost), np.cumsum(random_non_ood_score)/indices, label = 'Random')
    ax[0][1].set_title('EmbedLLM non-ood tasks')
    ax[0][1].set_ylim([0, 0.9])
    for m in selected_models:
        ax[0][1].scatter(single_models_non_ood[m][1], single_models_non_ood[m][0])
    ax[0][1].scatter(single_models_non_ood['oracle'][1], single_models_non_ood['oracle'][0], marker = '*', c = [223/255, 122/255, 94/255], label = 'oracle')
    




    selected_models = load_data('datasets/select_models_datasets.json')['routerbench']['selected_models']
    selected_datasets = load_data('datasets/select_models_datasets.json')['routerbench']['selected_datasets']
    test_data = load_data('datasets/RouterBench/routerbench_5shot-test.csv')
    bert_labels = load_data('output/routerbench/bert/bert-2025-04-13 16:57:36/evaluation-step-150-bert-False.json')
    bert_scores = bert_labels['scores']
    bert_costs = bert_labels['costs']
    bert_non_ood_score = [s for d, s in zip(test_data['dataset'].tolist(), bert_scores) if d in selected_datasets]
    bert_non_ood_cost = [c for d, c in zip(test_data['dataset'].tolist(), bert_costs) if d in selected_datasets]
    bert_non_ood_acc = round(sum(bert_non_ood_score)/len(bert_non_ood_score), 4)
    bert_ood_acc = round(sum(bert_scores)/len(bert_scores), 4)
    
    router_labels = load_data('output/routerbench/cluster4/2025-04-13 14:01:29/evaluation-step-200-newmodels-False.json')
    router_scores = router_labels['scores']
    router_costs = router_labels['costs']
    router_non_ood_score = [s for d, s in zip(test_data['dataset'].tolist(), router_scores) if d in selected_datasets]
    router_non_ood_cost = [c for d, c in zip(test_data['dataset'].tolist(), router_costs) if d in selected_datasets]
    router_non_ood_acc = round(sum(router_non_ood_score)/len(router_non_ood_score), 4)
    router_ood_acc = round(sum(router_scores)/len(router_scores), 4)
    
    qwen_labels = load_data('output/routerbench/qwen2.5/bert-2025-04-13 17:58:41/evaluation-step-350-bert-False.json')
    qwen_scores = qwen_labels['scores']
    qwen_costs = qwen_labels['costs']
    qwen_non_ood_score = [s for d, s in zip(test_data['dataset'].tolist(), qwen_scores) if d in selected_datasets]
    qwen_non_ood_cost = [c for d, c in zip(test_data['dataset'].tolist(), qwen_costs) if d in selected_datasets]
    qwen_non_ood_acc = round(sum(qwen_non_ood_score)/len(qwen_non_ood_score), 4)
    qwen_ood_acc = round(sum(qwen_scores)/len(qwen_scores), 4)
    
    knn_labels = load_data('output/routerbench/knn-result.json')
    knn_scores = knn_labels['score']
    knn_costs = knn_labels['cost']
    knn_non_ood_score = [s for d, s in zip(test_data['dataset'].tolist(), knn_scores) if d in selected_datasets]
    knn_non_ood_cost = [c for d, c in zip(test_data['dataset'].tolist(), knn_costs) if d in selected_datasets]
    knn_non_ood_acc = round(sum(knn_non_ood_score)/len(knn_non_ood_score), 4)
    knn_ood_acc = round(sum(knn_scores)/len(knn_scores), 4)
    
    random_labels = load_data('output/routerbench/random-result.json')
    random_scores = random_labels['score']
    random_costs = random_labels['cost']
    random_non_ood_score = [s for d, s in zip(test_data['dataset'].tolist(), random_scores) if d in selected_datasets]
    random_non_ood_cost = [c for d, c in zip(test_data['dataset'].tolist(), random_costs) if d in selected_datasets]
    random_non_ood_acc = round(sum(random_non_ood_score)/len(random_non_ood_score), 4)
    random_ood_acc = round(sum(random_scores)/len(random_scores), 4)
    
    single_models_non_ood = load_data('datasets/RouterBench/model_perf_cost-test-selected_datasets.json')
    single_models = load_data('datasets/RouterBench/model_perf_cost-test.json')
    
    print("Score: --routerbench")
    print(f"Router non-ood: {router_non_ood_acc}, cost: {sum(router_non_ood_cost)}, ood: {router_ood_acc}, cost: {sum(router_costs)}")
    print(f"qwen non-ood: {qwen_non_ood_acc}, cost: {sum(qwen_non_ood_cost)}, ood: {qwen_ood_acc}, cost: {sum(qwen_costs)}")
    print(f"Bert non-ood: {bert_non_ood_acc}, cost: {sum(bert_non_ood_cost)}, ood: {bert_ood_acc}, cost: {sum(bert_costs)}")
    print(f"knn non-ood: {knn_non_ood_acc}, cost: {sum(knn_non_ood_cost)}, ood: {knn_ood_acc}, cost: {sum(knn_costs)}")
    print(f"random non-ood: {random_non_ood_acc}, cost: {sum(random_non_ood_cost)}, ood: {random_ood_acc}, cost: {sum(random_costs)}")
    for m in selected_models  + ['oracle']:
        print(f"{m}: ood: {single_models[m][0]}, cost: {single_models[m][1]}, non-ood: {single_models_non_ood[m][0]}, cost: {single_models_non_ood[m][1]}")
    print('\n')
    indices = np.arange(1, len(test_data) + 1)
    ax[1][0].plot(np.cumsum(router_costs), np.cumsum(router_scores)/indices, label = 'Router')
    ax[1][0].plot(np.cumsum(bert_costs), np.cumsum(bert_scores)/indices, label = 'Bert')
    ax[1][0].plot(np.cumsum(qwen_costs), np.cumsum(qwen_scores)/indices, label = 'Qwen2.5-0.5B')
    ax[1][0].plot(np.cumsum(knn_costs), np.cumsum(knn_scores)/indices, label = 'Knn')
    ax[1][0].plot(np.cumsum(random_costs), np.cumsum(random_scores)/indices, label = 'Random')
    ax[1][0].set_ylabel('Accuracy')
    ax[1][0].set_xlabel('Cost')
    ax[1][0].set_title('RouterBench all tasks')
    ax[1][0].set_ylim([0, 1])
    for m in selected_models:
        ax[1][0].scatter(single_models[m][1], single_models[m][0])
    ax[1][0].scatter(single_models['oracle'][1], single_models['oracle'][0], marker = '*', c = [223/255, 122/255, 94/255])
    
    indices = np.arange(1, len(random_non_ood_score) + 1)
    ax[1][1].plot(np.cumsum(router_non_ood_cost), np.cumsum(router_non_ood_score)/indices, label = 'Router')
    ax[1][1].plot(np.cumsum(bert_non_ood_cost), np.cumsum(bert_non_ood_score)/indices, label = 'Bert')
    ax[1][1].plot(np.cumsum(qwen_non_ood_cost), np.cumsum(qwen_non_ood_score)/indices, label = 'Qwen2.5-0.5B')
    ax[1][1].plot(np.cumsum(knn_non_ood_cost), np.cumsum(knn_non_ood_score)/indices, label = 'Knn')
    ax[1][1].plot(np.cumsum(random_non_ood_cost), np.cumsum(random_non_ood_score)/indices, label = 'Random')
    ax[1][1].set_ylim([0, 1])
    ax[1][1].set_xlabel('Cost')
    ax[1][1].set_title('RouterBench non-ood tasks')
    for m in selected_models:
        ax[1][1].scatter(single_models_non_ood[m][1], single_models_non_ood[m][0])
    ax[1][1].scatter(single_models_non_ood['oracle'][1], single_models_non_ood['oracle'][0], marker = '*', c = [223/255, 122/255, 94/255], label = 'oracle')
    
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('figure/all-non-ood.pdf', dpi = 300)
    plt.show()
    


def embed_single_dataset():
    fig, ax = plt.subplots(4,2, figsize = (8, 4))
    selected_datasets = load_data('datasets/select_models_datasets.json')['embedllm1']['selected_datasets']
    # selected_datasets = load_data('datasets/select_models_datasets.json')['embedllm1']['new_datasets']
    selected_models = load_data('datasets/select_models_datasets.json')['embedllm1']['selected_models']
    
    m, n = 0, 0
    for i, dataset in enumerate(selected_datasets):
        test_data = load_data('datasets/embedLLM/embed-test.csv')
        bert_labels = load_data('output/embedllm/bert/bert-2025-04-13 17:20:51/evaluation-step-400-bert-False.json')
        bert_scores = bert_labels['scores']
        bert_costs = bert_labels['costs']
        bert_score = [s for d, s in zip(test_data['dataset'].tolist(), bert_scores) if d in dataset]
        bert_cost = [c for d, c in zip(test_data['dataset'].tolist(), bert_costs) if d in dataset]
        bert_acc = round(sum(bert_score)/len(bert_score), 4)
        # bert_ood_acc = round(sum(bert_scores)/len(bert_scores), 4)
        
        router_labels = load_data('output/embedllm/cluster2/2025-04-13 14:19:31/evaluation-step-150-newmodels-False.json')
        router_labels1 = load_data('output/embedllm/cluster3/2025-04-13 14:41:38/evaluation-step-500-newmodels-False.json')
        router_scores = router_labels1['scores']
        router_costs = router_labels['costs']
        router_score = [s for d, s in zip(test_data['dataset'].tolist(), router_scores) if d in dataset]
        router_cost = [c for d, c in zip(test_data['dataset'].tolist(), router_costs) if d in dataset]
        router_acc = round(sum(router_score)/len(router_score), 4)
        # router_ood_acc = round(sum(router_scores)/len(router_scores), 4)
        
        qwen_labels = load_data('output/embedllm/qwen2.5/bert-2025-04-13 17:37:49/evaluation-step-200-bert-False.json')
        qwen_scores = qwen_labels['scores']
        qwen_costs = qwen_labels['costs']
        qwen_score = [s for d, s in zip(test_data['dataset'].tolist(), qwen_scores) if d in dataset]
        qwen_cost = [c for d, c in zip(test_data['dataset'].tolist(), qwen_costs) if d in dataset]
        qwen_acc = round(sum(qwen_score)/len(qwen_score), 4)
        
        knn_labels = load_data('output/embedllm/knn-result.json')
        knn_scores = knn_labels['score']
        knn_costs = knn_labels['cost']
        knn_score = [s for d, s in zip(test_data['dataset'].tolist(), knn_scores) if d in dataset]
        knn_cost = [c for d, c in zip(test_data['dataset'].tolist(), knn_costs) if d in dataset]
        knn_acc = round(sum(knn_score)/len(knn_score), 4)
        
        random_labels = load_data('output/embedllm/random-result.json')
        random_scores = random_labels['score']
        random_costs = random_labels['cost']
        random_score = [s for d, s in zip(test_data['dataset'].tolist(), random_scores) if d in dataset]
        random_cost = [c for d, c in zip(test_data['dataset'].tolist(), random_costs) if d in dataset]
        random_acc = round(sum(random_score)/len(random_score), 4)
        
        single_models = load_data('datasets/embedLLM/model_perf_cost-test-single_datasets.json')
        
        print(f"Score: {dataset}")
        print(f"Router non-ood: {router_acc}, cost: {sum(router_cost)}")
        print(f"qwen non-ood: {qwen_acc}, cost: {sum(qwen_cost)}")
        print(f"Bert non-ood: {bert_acc}, cost: {sum(bert_cost)}")
        print(f"knn non-ood: {knn_acc}, cost: {sum(knn_cost)}")
        print(f"random non-ood: {random_acc}, cost: {sum(random_cost)}")
        m_scores = {}
        m_costs = {}
        for mm in selected_models  + ['oracle']:
            m_scores[mm] = sum(single_models[dataset][mm]['score'])/len(single_models[dataset][mm]['score'])
            m_costs[mm] = sum(single_models[dataset][mm]['cost'])
            print(f"{mm}: score: {m_scores[mm]}, cost: {m_costs[mm]}")
        print('\n')
        
        
        indices = np.arange(1, len(bert_score) + 1)
        ax[m][n].plot(np.cumsum(router_cost), np.cumsum(router_score)/indices[:len(router_score)], label = 'Router')
        ax[m][n].plot(np.cumsum(bert_cost), np.cumsum(bert_score)/indices[:len(bert_score)], label = 'Bert')
        ax[m][n].plot(np.cumsum(qwen_cost), np.cumsum(qwen_score)/indices[:len(qwen_score)], label = 'Qwen2.5-0.5B')
        ax[m][n].plot(np.cumsum(knn_cost), np.cumsum(knn_score)/indices[:len(knn_score)], label = 'Knn')
        ax[m][n].plot(np.cumsum(random_cost), np.cumsum(random_score)/indices[:len(random_score)], label = 'Random')
        ax[m][n].set_ylabel('Accuracy')
        ax[m][n].set_title(f'{dataset}')
        ax[m][n].set_ylim([-0.1, 1])

        for mm in selected_models:
            ax[m][n].scatter(m_costs[mm], m_scores[mm])
        ax[m][n].scatter(m_costs['oracle'], m_scores['oracle'], marker = '*', c = [223/255, 122/255, 94/255])
        
        # ax[m].plot(np.cumsum(router_cost), np.cumsum(router_score)/indices[:len(router_score)], label = 'Router')
        # ax[m].plot(np.cumsum(bert_cost), np.cumsum(bert_score)/indices[:len(bert_score)], label = 'Bert')
        # ax[m].plot(np.cumsum(qwen_cost), np.cumsum(qwen_score)/indices[:len(qwen_score)], label = 'Qwen2.5-0.5B')
        # ax[m].plot(np.cumsum(knn_cost), np.cumsum(knn_score)/indices[:len(knn_score)], label = 'Knn')
        # ax[m].plot(np.cumsum(random_cost), np.cumsum(random_score)/indices[:len(random_score)], label = 'Random')
        # ax[m].set_ylabel('Accuracy')
        # ax[m].set_title(f'{dataset}')
        # ax[m].set_ylim([-0.1, 1])

        # for mm in selected_models:
        #     ax[m].scatter(m_costs[mm], m_scores[mm])
        # ax[m].scatter(m_costs['oracle'], m_scores['oracle'], marker = '*', c = [223/255, 122/255, 94/255])
        
        # m += 1
        n += 1
        if n == 2:
            m += 1
            n = 0
        
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('figure/embedllm-single-datasets-ood.pdf', dpi = 300)
    # plt.show()





def routerbench_single_dataset():
    fig, ax = plt.subplots(1,2, figsize = (8, 4))
    # selected_datasets = load_data('datasets/select_models_datasets.json')['routerbench']['selected_datasets']
    selected_datasets = load_data('datasets/select_models_datasets.json')['routerbench']['new_datasets']
    selected_models = load_data('datasets/select_models_datasets.json')['routerbench']['selected_models']

    
    m, n = 0, 0
    for i, dataset in enumerate(selected_datasets):
        test_data = load_data('datasets/RouterBench/routerbench_5shot-test.csv')
        bert_labels = load_data('output/routerbench/bert/bert-2025-04-13 16:57:36/evaluation-step-150-bert-False.json')
        bert_scores = bert_labels['scores']
        bert_costs = bert_labels['costs']
        bert_score = [s for d, s in zip(test_data['dataset'].tolist(), bert_scores) if d in dataset]
        bert_cost = [c for d, c in zip(test_data['dataset'].tolist(), bert_costs) if d in dataset]
        bert_acc = round(sum(bert_score)/len(bert_score), 4)
        # bert_ood_acc = round(sum(bert_scores)/len(bert_scores), 4)
        
        router_labels = load_data('output/routerbench/cluster4/2025-04-13 14:01:29/evaluation-step-200-newmodels-False.json')
        router_scores = router_labels['scores']
        router_costs = router_labels['costs']
        router_score = [s for d, s in zip(test_data['dataset'].tolist(), router_scores) if d in dataset]
        router_cost = [c for d, c in zip(test_data['dataset'].tolist(), router_costs) if d in dataset]
        router_acc = round(sum(router_score)/len(router_score), 4)
        # router_ood_acc = round(sum(router_scores)/len(router_scores), 4)
        
        qwen_labels = load_data('output/routerbench/qwen2.5/bert-2025-04-13 17:58:41/evaluation-step-350-bert-False.json')
        qwen_scores = qwen_labels['scores']
        qwen_costs = qwen_labels['costs']
        qwen_score = [s for d, s in zip(test_data['dataset'].tolist(), qwen_scores) if d in dataset]
        qwen_cost = [c for d, c in zip(test_data['dataset'].tolist(), qwen_costs) if d in dataset]
        qwen_acc = round(sum(qwen_score)/len(qwen_score), 4)
        
        knn_labels = load_data('output/routerbench/knn-result.json')
        knn_scores = knn_labels['score']
        knn_costs = knn_labels['cost']
        knn_score = [s for d, s in zip(test_data['dataset'].tolist(), knn_scores) if d in dataset]
        knn_cost = [c for d, c in zip(test_data['dataset'].tolist(), knn_costs) if d in dataset]
        knn_acc = round(sum(knn_score)/len(knn_score), 4)
        
        random_labels = load_data('output/routerbench/random-result.json')
        random_scores = random_labels['score']
        random_costs = random_labels['cost']
        random_score = [s for d, s in zip(test_data['dataset'].tolist(), random_scores) if d in dataset]
        random_cost = [c for d, c in zip(test_data['dataset'].tolist(), random_costs) if d in dataset]
        random_acc = round(sum(random_score)/len(random_score), 4)
        
        single_models = load_data('datasets/RouterBench/model_perf_cost-test-single_datasets.json')
        
        print(f"Score: {dataset}")
        print(f"Router non-ood: {router_acc}, cost: {sum(router_cost)}")
        print(f"qwen non-ood: {qwen_acc}, cost: {sum(qwen_cost)}")
        print(f"Bert non-ood: {bert_acc}, cost: {sum(bert_cost)}")
        print(f"knn non-ood: {knn_acc}, cost: {sum(knn_cost)}")
        print(f"random non-ood: {random_acc}, cost: {sum(random_cost)}")
        m_scores = {}
        m_costs = {}
        for mm in selected_models  + ['oracle']:
            m_scores[mm] = sum(single_models[dataset][mm]['score'])/len(single_models[dataset][mm]['score'])
            m_costs[mm] = sum(single_models[dataset][mm]['cost'])
            print(f"{mm}: score: {m_scores[mm]}, cost: {m_costs[mm]}")
        print('\n')
        
        indices = np.arange(1, len(bert_score) + 1)
        # ax[m][n].plot(np.cumsum(router_cost), np.cumsum(router_score)/indices[:len(router_score)], label = 'Router')
        # ax[m][n].plot(np.cumsum(bert_cost), np.cumsum(bert_score)/indices[:len(bert_score)], label = 'Bert')
        # ax[m][n].plot(np.cumsum(qwen_cost), np.cumsum(qwen_score)/indices[:len(qwen_score)], label = 'Qwen2.5-0.5B')
        # ax[m][n].plot(np.cumsum(knn_cost), np.cumsum(knn_score)/indices[:len(knn_score)], label = 'Knn')
        # ax[m][n].plot(np.cumsum(random_cost), np.cumsum(random_score)/indices[:len(random_score)], label = 'Random')
        # ax[m][n].set_ylabel('Accuracy')
        # ax[m][n].set_title(f'{dataset}')
        # ax[m][n].set_ylim([-0.1, 1])

        # for mm in selected_models:
        #     ax[m][n].scatter(m_costs[mm], m_scores[mm])
        # ax[m][n].scatter(m_costs['oracle'], m_scores['oracle'], marker = '*', c = [223/255, 122/255, 94/255])
        
        ax[m].plot(np.cumsum(router_cost), np.cumsum(router_score)/indices[:len(router_score)], label = 'Router')
        ax[m].plot(np.cumsum(bert_cost), np.cumsum(bert_score)/indices[:len(bert_score)], label = 'Bert')
        ax[m].plot(np.cumsum(qwen_cost), np.cumsum(qwen_score)/indices[:len(qwen_score)], label = 'Qwen2.5-0.5B')
        ax[m].plot(np.cumsum(knn_cost), np.cumsum(knn_score)/indices[:len(knn_score)], label = 'Knn')
        ax[m].plot(np.cumsum(random_cost), np.cumsum(random_score)/indices[:len(random_score)], label = 'Random')
        ax[m].set_ylabel('Accuracy')
        ax[m].set_title(f'{dataset}')
        ax[m].set_ylim([-0.1, 1.05])

        for mm in selected_models:
            ax[m].scatter(m_costs[mm], m_scores[mm])
        ax[m].scatter(m_costs['oracle'], m_scores['oracle'], marker = '*', c = [223/255, 122/255, 94/255])
        
        # n += 1
        # if n == 2:
        #     m += 1
        #     n = 0
        m += 1
        
    plt.legend()
    plt.tight_layout()
    plt.savefig('figure/routerbench-single-datasets-ood.pdf', dpi = 300)
    plt.show()


def cluster():
    fig, ax = plt.subplots(1,2, figsize = (8,4))
    selected_datasets = load_data('datasets/select_models_datasets.json')['routerbench']['selected_datasets']
    test_data = load_data('datasets/RouterBench/routerbench_5shot-test.csv')
    cluster_path = ['2025-04-13 13:04:09', '2025-04-13 13:29:09', '2025-04-13 14:01:29', '2025-04-13 14:10:04']
    shape = ['o', 's', '^', 'D', '*', '+', 'x']
    step = 50
    cluster = 2
    for path in cluster_path:
        router_cluster = load_data(f"output/routerbench/cluster{cluster}/{cluster_path[cluster - 2]}/evaluation-step-{step}-newmodels-False.json")
        for dataset in selected_datasets:
            router_scores = router_cluster['scores']
            router_costs = router_cluster['costs']
            router_score = [s for d, s in zip(test_data['dataset'].tolist(), router_scores) if d in dataset]
            router_cost = [c for d, c in zip(test_data['dataset'].tolist(), router_costs) if d in dataset]
            router_acc = round(sum(router_score)/len(router_score), 4)
            
            ax[0].scatter(sum(router_cost), router_acc, marker = shape[cluster - 2])
            ax[0].text(sum(router_cost) - 0.005, router_acc + 0.005, s = f'{cluster}')
        cluster += 1
    
    plt.show()
            

            
            
        
    
    
    

if __name__ == "__main__":
    # all()
    # embed_single_dataset()
    routerbench_single_dataset()
    # cluster()