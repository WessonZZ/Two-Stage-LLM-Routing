# -*- coding: utf-8 -*-
# @Time    : 2025/4/13 11:38
# @Author  : Wesson
# @FileName: bert


from argparse import ArgumentParser, ArgumentTypeError
import json
import os
from tqdm import tqdm
import time
from collections import Counter

import pandas as pd
import numpy as np
import random
from typing import OrderedDict
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from Dataclass import RouterData


def str_to_bool(value):
    if value.lower() in ("true", "yes", "t", "y", "1"):
        return True
    elif value.lower() in ("false", "no", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")

def parse_config():
    parser = ArgumentParser()
    parser.add_argument("--model_path",           type = str, default = "microsoft/mdeberta-v3-base")
    parser.add_argument("--data_path",            type = str, default = 'datasets/embedLLM/embed-train.csv')
    parser.add_argument("--cluster_path",         type = str, default = 'datasets/embedLLM/embed-train.csv')
    parser.add_argument("--eval_path",            type = str, default = 'datasets/embedLLM/embed-val.csv')
    parser.add_argument("--max_seq_length",       type = int, default = 512)
    parser.add_argument("--dim",                  type = int, default = 768)
    parser.add_argument("--n_models",             type = int, default = 9)

    parser.add_argument("--is_ood",               type = str_to_bool, default = False)
    parser.add_argument("--is_newmodels",         type = str_to_bool, default = False)
    parser.add_argument("--batch_size",           type = int, default = 64)
    parser.add_argument("--eval_size",            type = int, default = 100)
    parser.add_argument("--epochs",               type = int, default = 10)
    parser.add_argument("--lr",                   type = float, default = 5e-5)
    parser.add_argument("--gradient_accumulation",type = int, default = 1)
    parser.add_argument("--warmup_rate",          type = float, default = 0.1)
    parser.add_argument("--use_scheduler",        type = str_to_bool, default = True)
    parser.add_argument("--scheduler",            type = str, default = 'cosine')
    parser.add_argument("--eval_steps",           type = int, default = 200)
    parser.add_argument("--seed",                 type = int, default = 42)
    parser.add_argument("--log_steps",            type = int, default = 5)
    parser.add_argument("--save_steps",           type = int, default = 1000)
    parser.add_argument("--save_folder",          type = str, default = 'outputs')
    parser.add_argument("--checkpoint_dir",       type = str, default = 'ck_dir')
    
    return parser.parse_args()



def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        param_count = 1
        for size in param.shape:
            param_count *= size
        total_params += param_count
    return total_params


class RouterBert(nn.Module):
    def __init__(self, backbone, config, device):
        super().__init__()
        self.backbone = backbone
        self.n_models = config.n_models
        self.last_hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.last_hidden_size, self.n_models, bias = False))
        ]))
        
    def forward(self, **inputs):
        x = self.backbone(**inputs)
        # We used the first token as classifier token.
        hidden_state = x['last_hidden_state'][:, -1, :]
        output = self.classifier(hidden_state)
        return output
    
def load_model(model_path):
    # "microsoft/mdeberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side = 'left', padding = True)
    model = AutoModel.from_pretrained(model_path)
    
    return tokenizer, model

def get_scheduler(optimizer, config, training_step):
    warmup_steps = int(config.warmup_rate * training_step)
    num_training_steps = (config.epochs + config.warmup_rate) * training_step
    if config.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = warmup_steps,
                                                    num_training_steps= num_training_steps)
    elif config.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = warmup_steps,
                                                    num_training_steps = num_training_steps)
    else:
        scheduler = None
    return scheduler


def get_loss(logit, labels):
    
    return nn.CrossEntropyLoss()(logit, labels)

def evaluation(model, config, tokenizer, device, save_path, steps, is_ood, is_newmodels):
    print("### Eval start...")
    model.eval()
    evaldata = RouterData(config.eval_path, config.cluster_path, config.max_seq_length, tokenizer, is_ood, is_newmodels)
    evalloader = DataLoader(evaldata, batch_size = config.batch_size, shuffle = False)

    score = []
    cost = []
    selected_model = []
    top_score = 0
    with torch.no_grad():
        for idx_, batch in tqdm(enumerate(evalloader), total = len(evalloader), desc = 'Sample'):
            inputs, cluster_pos, cluster_neg, model_index, model_labels, model_costs,_ = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}

            cluster_pos = cluster_pos.to(device)
            cluster_neg = cluster_neg.to(device)
            model_index = model_index.to(device)
            # print(cluster_pos)

            logits = model.forward(**inputs)
            
            pred = torch.argmax(logits, dim = -1).tolist()
            
            for cc, c in enumerate(pred):
                
                best_model_order = pred[cc]
                model_index_ = model_index[cc, :].tolist()
                
                model_labels_ = model_labels[cc, :].tolist()
                
                costs = model_costs[cc, :].tolist()
                
                assert len(model_index_) == len(model_labels_)
                # assert len(pos_model_index_) == len(pos_model_labels_)
                
                index_labels = {str(int(i)): l for i, l in zip(model_index_, model_labels_)}
                index_costs = {str(int(i)): c for i, c in zip(model_index_, costs)}

                score.append(index_labels[str(int(best_model_order))])
                cost.append(index_costs[str(int(best_model_order))])
                selected_model.append(int(best_model_order))
                top_score += max(model_labels_)
            
            samples_eval = (idx_ + 1) * config.batch_size
            if samples_eval >= config.eval_size:
                break
    score_ = sum(score)
    cost_ = sum(cost)
    print("### Eval ended")
    print(f"Total {samples_eval} samples, score: {score_}/{len(evaldata)} = {round(score_/len(evaldata),4)}, cost: {cost_}\n")
    with open(f"{save_path}/evaluation-step-{steps}-bert-{is_newmodels}.json", 'w', encoding='utf-8') as f:
        json.dump({'final_score': score_, 'final_cost': cost_, 'top_score': len(evaldata), 'scores': score, 'costs': cost, 'select_m': selected_model}, f, indent = 4)
    print("Saved.")
    
    return score_/len(evaldata)
    
    

def train(config):
    timestamp = time.time()
    local_time = time.localtime(timestamp)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    local_save_path = os.path.join(config.save_folder, f"bert-{formatted_time}")
    os.makedirs(local_save_path, exist_ok=True)
    
    config_dict = vars(config)
    with open(local_save_path + '/config.json', 'w', encoding = 'utf-8') as f:
        json.dump(config_dict, f, indent = 4)
        
    setup_seed(config.seed)
    tokenizer, model = load_model(config.model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    routerdata = RouterData(config.data_path, config.cluster_path, config.max_seq_length, tokenizer, config.is_ood, config.is_newmodels)
    # routerloader1 = DataLoader(routerdata, batch_size = config.batch_size, shuffle = True, worker_init_fn= seed_worker)
    routerloader = DataLoader(routerdata, batch_size = config.batch_size, shuffle = True, worker_init_fn= seed_worker)
    
    routermodel = RouterBert(model, config, device)
    routermodel.to(device)
    
    paras_num = count_parameters(model)
    print(f"Total params: {paras_num}")
    
    optimizer = torch.optim.AdamW(routermodel.parameters(), lr=config.lr)
    
    
    if config.use_scheduler:
        scheduler = get_scheduler(optimizer, config, len(routerloader))
        # scheduler2 = get_scheduler(optimizer2, config, len(routerloader))
    scaler = GradScaler()
    print("Training start...")

    for epoch in tqdm(range(config.epochs)):
        steps = 0

        for batch in routerloader:
            routermodel.train()
            routermodel.train()
            optimizer.zero_grad()
            
            inputs, _, _, model_index, _, _, _ = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}

            model_index = model_index.to(device)
            label = model_index[:, 0].view(-1).long()

            with autocast():
                logits = routermodel.forward(**inputs).float()
                loss = get_loss(logits, label)

            scaler.scale(loss).backward()
            
            steps += 1
            if (steps + 1) % config.gradient_accumulation == 0:
                # 只在这里 unscale + clip + step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(routermodel.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()

                if config.use_scheduler:
                    scheduler.step()
                
            if (steps + 1) % config.log_steps == 0:
                print(f"##Epoch {epoch}  ## Step {steps + 1}, projector-lr: {optimizer.param_groups[0]['lr']}, loss: {loss.item()}")
            
            if (steps + 1) % config.eval_steps == 0:
                _ = evaluation(routermodel, config, tokenizer, device, config.save_folder + f"/bert-{formatted_time}", steps + 1, True, config.is_newmodels)
            


def main():
    config = parse_config()
    os.makedirs(config.save_folder, exist_ok=True)
    
    train(config)
            
if __name__ == "__main__":
    
    main()    
            