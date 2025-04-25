# -*- coding: utf-8 -*-
# @Time    : 2025/4/7 20:00
# @Author  : Wesson
# @FileName: data_process

import pandas as pd
import numpy as np
from collections import Counter
import random
import ast
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Chalkboard SE'
# plt.rcParams['font.size'] = 10


def routerBench(path):
    
    few_shot = pd.DataFrame(pd.read_pickle(path))
    few_shot = few_shot.fillna(0)
    
    data_set = []
    for i in range(len(few_shot)):
        if few_shot.loc[i, 'eval_name'] in ['grade-school-math']:
            data_set.append('gsm8k')
        elif 'mmlu' in few_shot.loc[i, 'eval_name'].lower():
            data_set.append('mmlu')
        elif 'mtbench' in few_shot.loc[i, 'eval_name']:
            data_set.append('mtbench')
        elif few_shot.loc[i, 'eval_name'] in ['winogrande']:
            data_set.append('winogrande')
        elif few_shot.loc[i, 'eval_name'] in ['mbpp']:
            data_set.append('mbpp')
        elif few_shot.loc[i, 'eval_name'] in ['arc-challenge']:
            data_set.append('arc-c')
        elif few_shot.loc[i, 'eval_name'] in ['hellaswag']:
            data_set.append('hellaswag')
        else:
            data_set.append('rag')
            
    few_shot['dataset'] = data_set
    # few_shot.to_csv('datasets/RouterBench/routerbench_5shot.csv', index= False)
    
    def convert_str(prompt):
        prompt = ast.literal_eval(prompt)
        assert isinstance(prompt, list), print("Type Error")
        if len(prompt) > 1:
            return " ".join(prompt)
        elif len(prompt) == 1:
            return prompt[0]

    few_shot['prompt'] = few_shot['prompt'].apply(convert_str)
    
    models = few_shot.columns.tolist()[3:14]
    
    def construct_pos_neg(sample):
        # print(sample)
        score_cost_df = pd.DataFrame({'models': models})
        score_cost_df = score_cost_df.fillna(0)
        scores = [sample[m] for m in models]
        score_cost_df['score'] = scores
        score_cost_df['cost'] = [sample[m+"|total_cost"] for m in models]
        score_cost_df = score_cost_df.sort_values(by = ['score', 'cost'], ascending= [False, True])
        
        pos = score_cost_df.loc[score_cost_df['score'] > 0.05, 'models'].tolist()
        neg = score_cost_df.loc[score_cost_df['score'] <= 0.05, 'models'].tolist()
        
        return pos, neg
    
    Pos, Neg = [], []
    for i in tqdm(range(len(few_shot)), desc = "Sample", total = len(few_shot)):
        sample = few_shot.iloc[i, :]
        # print(sample)
        pos, neg = construct_pos_neg(sample)
        Pos.append(pos)
        Neg.append(neg)
        
    few_shot['pos'] = Pos
    few_shot['neg'] = Neg
    
    selected_columns = [c for c in few_shot.columns.tolist() if 'model_response' not in c]
    few_shot = few_shot.loc[:, selected_columns]
    
    dataset_ls = list(set(data_set))
    sample_val = {'arc-c': 200, 'gsm8k': 300, 'hellaswag': 300, 'mbpp': 100, 'mmlu': 300, 'mtbench': 20, 'rag': 300, 'winogrande': 200}
    sample_test = {'arc-c': 300, 'gsm8k': 300, 'hellaswag': 300, 'mbpp': 100, 'mmlu': 300, 'mtbench': 20, 'rag': 300, 'winogrande': 300}


    few_shot_val = pd.DataFrame([])
    few_shot_test = pd.DataFrame([])
    few_shot_train = pd.DataFrame([])
    for data in dataset_ls:
        data_df = few_shot.loc[few_shot['dataset'] == data, :]
        data_sample = random.sample(range(len(data_df)), len(data_df))
        data_val = data_sample[:sample_val[data]]
        data_test = data_sample[sample_val[data]:sample_val[data] + sample_test[data]]
        data_train = data_sample[sample_val[data] + sample_test[data]:]
        few_shot_val = pd.concat([few_shot_val, data_df.iloc[data_val, :]])
        few_shot_test = pd.concat([few_shot_test, data_df.iloc[data_test, :]])
        few_shot_train = pd.concat([few_shot_train, data_df.iloc[data_train, :]])
        
    few_shot_val.to_csv('datasets/RouterBench/routerbench_5shot-val.csv', index= False)
    few_shot_test.to_csv('datasets/RouterBench/routerbench_5shot-test.csv', index= False)
    few_shot_train.to_csv('datasets/RouterBench/routerbench_5shot-train.csv', index= False)
    
    # models = ['WizardLM/WizardLM-13B-V1.2', 'claude-instant-v1', 'claude-v1', 'claude-v2', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'meta/code-llama-instruct-34b-chat', 'meta/llama-2-70b-chat', 'mistralai/mistral-7b-chat', 'mistralai/mixtral-8x7b-chat', 'zero-one-ai/Yi-34B-Chat']
    cost_train = few_shot_train.loc[:,[m+"|total_cost" for m in models]].mean().values
    perf_cost = pd.DataFrame({'models': models, 'avg_cost': cost_train})
    dataset_perf_train = few_shot_train.groupby(['dataset'])[models].mean().T.reset_index()
    perf_cost = pd.concat([perf_cost, dataset_perf_train.iloc[:, 1:]], axis = 1)
    
    perf_cost.to_csv('datasets/RouterBench/routerbench_5shot-perf-cost.csv', index= False)
    
    print('RouterBench processed finished!')


def embedLLM():
    
    embed_val = pd.read_csv('datasets/embedLLM/val.csv')
    embed_test = pd.read_csv('datasets/embedLLM/test.csv')
    embed_train = pd.read_csv('datasets/embedLLM/new_train_subset_20k.csv').iloc[:, 1:]
    
    def map_category(category):
        if 'gsm8k' in category:
            return 'gsm8k'
        elif 'mmlu' in category.lower():
            return 'mmlu'
        elif 'gpqa' in category:
            return 'gpqa'
        elif 'truthfulqa' in category:
            return 'truthfulqa'
        elif 'mathqa' in category:
            return 'mathqa'
        elif 'asdiv' in category:
            return 'asdiv'
        elif 'social_iqa' in category:
            return 'socialqa'
        elif 'medmcqa' in category:
            return 'medmcqa'
        elif 'piqa' in category:
            return 'piqa'
        else:
            return 'logiqa'
        
    def map_model_name(model_name):
        return model_name.replace('__', '/')

    for data_split in [embed_test, embed_train, embed_val]:

        data_split['dataset'] = data_split['category'].apply(map_category)
        data_split['model_name'] = data_split['model_name'].apply(map_model_name)
        
        data_split = data_split.drop(columns = ['model_id', 'category_id']).drop_duplicates()

    pivot_val = embed_val.pivot_table(index= ['prompt_id', 'category', 'prompt', 'dataset'], columns = 'model_name', values = 'label', aggfunc = 'first').reset_index()
    pivot_test = embed_test.pivot_table(index= ['prompt_id', 'category', 'prompt', 'dataset'], columns = 'model_name', values = 'label', aggfunc = 'first').reset_index()
    pivot_train = embed_train.pivot_table(index= ['prompt_id', 'category', 'prompt', 'dataset'], columns = 'model_name', values = 'label', aggfunc = 'first').reset_index()
    print("Pivot finished.")
    
    price = pd.read_csv('datasets/embedLLM/llm_pricing.csv')
    models = price['Model Name'].tolist()
    price_dict = dict(zip(price['Model Name'], price['Price']))
    
    def construct_pos_neg(sample_dict):
        # print(sample)
        scores = np.array([sample_dict[m] for m in models])
        costs = np.array([price_dict[m] for m in models])

        assert len(scores) == 112
        
        sort_idx = np.lexsort((costs, -scores))
        sorted_models = np.array(models)[sort_idx]
        sorted_scores = scores[sort_idx]
        
        pos = sorted_models[sorted_scores > 0].tolist()
        neg = sorted_models[sorted_scores == 0].tolist()
        return pos, neg
    
    for data_split in [pivot_val, pivot_test, pivot_train]:
        Pos, Neg = [], []
        prompt_id = []
        for i, row in tqdm(enumerate(data_split.itertuples(index=False, name = None)), total=len(data_split), desc="Sample"):
            sample_dict = dict(zip(data_split.columns, row))  # 转为字典更快
            pos, neg = construct_pos_neg(sample_dict)
            Pos.append(pos)
            Neg.append(neg)
            prompt_id.append(i)
        data_split['pos'] = Pos
        data_split['neg'] = Neg
        data_split['prompt_id'] = prompt_id
    
    pivot_test.to_csv('datasets/embedLLM/embed-test.csv', index= False)
    pivot_val.to_csv('datasets/embedLLM/embed-val.csv', index= False)
    pivot_train.to_csv('datasets/embedLLM/embed-train.csv', index= False)
    
    perf_train_T = embed_train.pivot_table(values = 'label', index = 'dataset', columns = 'model_name', aggfunc = 'mean').T.reset_index()
    # perf_train_all = embed_train.pivot_table(values = 'label', columns = 'model_name', aggfunc = 'mean')
    
    perf_train_T = perf_train_T.merge(price.drop(columns=['Parameters', 'Pricing Basis']), how='left', left_on = 'model_name', right_on='Model Name').drop(columns='Model Name')
    perf_train_T = perf_train_T.rename(columns={'model_name': 'models', 'Price': 'avg_cost'})
    
    perf_train_T.to_csv('datasets/embedLLM/embedLLM-perf-cost.csv', index= False)
    print('EmbedLLM processed finished!')



if __name__ == "__main__":
    
    routerBench(path = 'datasets/RouterBench/routerbench_5shot.pkl')
    embedLLM()
