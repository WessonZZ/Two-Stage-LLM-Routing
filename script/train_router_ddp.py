# -*- coding: utf-8 -*-
# @Time    : 2025/4/8 21:31
# @Author  : Wesson
# @FileName: train_router

from argparse import ArgumentParser, ArgumentTypeError
import json
import os
from tqdm import tqdm
import time
from collections import Counter

import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, DebertaV2Model, DebertaV2Tokenizer, AutoModel
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
    parser.add_argument("--embedding_path",       type = str, default = "None")
    parser.add_argument("--projector_path",       type = str, default = "None")
    parser.add_argument("--max_seq_length",       type = int, default = 512)
    parser.add_argument("--dim",                  type = int, default = 768)

    parser.add_argument("--is_ood",               type = str_to_bool, default = False)
    parser.add_argument("--is_newmodels",         type = str_to_bool, default = False)
    parser.add_argument("--simi",                 type = str, default = 'cos')
    parser.add_argument("--tau",                  type = float, default = 1.0)
    parser.add_argument("--margin",               type = float, default = 0.3)
    parser.add_argument("--alpha",                type = float, default = 0.5)
    parser.add_argument("--beta",                 type = float, default = 1.0)
    parser.add_argument("--batch_size",           type = int, default = 64)
    parser.add_argument("--eval_size",            type = int, default = 100)
    parser.add_argument("--epochs",               type = int, default = 10)
    parser.add_argument("--lr",                   type = float, default = 5e-5)
    parser.add_argument("--gradient_accumulation",type = int, default = 1)
    parser.add_argument("--warmup_rate",          type = float, default = 0.1)
    parser.add_argument("--use_scheduler",        type = str_to_bool, default = True)
    parser.add_argument("--scheduler",            type = str, default = 'cosine')
    parser.add_argument("--c_steps",              type = int, default = 50)
    parser.add_argument("--eval_steps",           type = int, default = 200)
    parser.add_argument("--seed",                 type = int, default = 42)
    parser.add_argument("--log_steps",            type = int, default = 5)
    parser.add_argument("--save_steps",           type = int, default = 1000)
    parser.add_argument("--save_folder",          type = str, default = 'outputs')
    parser.add_argument("--checkpoint_dir",       type = str, default = 'ck_dir')
    parser.add_argument("--local_rank",           type=int, default=-1, help="Local rank for distributed training")
    
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


class RouterModule(nn.Module):
    def __init__(self, backbone, config, device):
        super().__init__()
        self.data_path = config.data_path
        self.cluster_path = config.cluster_path
        self.embedding_path = config.embedding_path
        self.data_name = 'embedllm' if 'embedllm' in self.data_path.lower() else 'routerbench'
        self.backbone = backbone
        self.device = device
        self.is_ood = config.is_ood
        self.simi = config.simi
        self.tau = config.tau 
        self.margin = torch.tensor(config.margin).to(self.device)
        self.dim = config.dim
        self._select_model_dataset()
        self.model_clusters = self.load_clusters()
        
        self.model_embeddings = self._load_embeddings()
        # mean, std = 0, 0.78
        self.projector = self._load_projector().to(self.device) #conversion matrix
        self.cluster_embedding = self._load_cluster_embedding()

    def _load_embeddings(self):
        #init for model embeddings
        if os.path.exists(self.embedding_path):
            embeddings = self.load_data(self.embedding_path)
        else:
            embeddings = ''
        if 'model_embedding' in embeddings:
            embeddings = embeddings['model_embedding']
        else:
            raw_embeds = torch.randn(self.n_models, self.dim)
            nn.init.normal_(raw_embeds, mean=0, std=0.78)
            embeddings = nn.ParameterDict({
                str(self.model_orders[name]): nn.Parameter(raw_embeds[i]).to(self.device)
                for i, name in enumerate(self.selected_models)
            })

        return embeddings
    
    
    def _load_projector(self):
        if os.path.exists(self.embedding_path):
            projector =  self.load_data(self.embedding_path)
        else:
            projector = ''
        if 'projector' in projector:
            projector = projector['projector']
        else:
            projector = nn.Parameter(torch.randn(self.dim, len(self.perf_dimensions)))
            nn.init.normal_(projector, mean=0, std = 1)

        return projector

    def _load_cluster_embedding(self):
        #init cluster embeddings
                #init for model embeddings
        if os.path.exists(self.embedding_path):
            embeddings = self.load_data(self.embedding_path)
        else:
            embeddings = ''
        if 'cluster_embedding' in embeddings:
            embeddings = embeddings['cluster_embedding']
        else:
            raw_embeds = torch.randn(self.n_clusters, self.dim)
            nn.init.normal_(raw_embeds, mean=0, std=0.78)
            embeddings = nn.ParameterDict({
                str(i): nn.Parameter(raw_embeds[i]).to(self.device)
                for i in range(self.n_clusters)
            })

        return embeddings


    def load_data(self, path):
        
        if path.endswith('csv'):
            data = pd.read_csv(path)
        elif path.endswith('json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = pd.DataFrame(json.load(f))
        else:
            pass

        return data
    
    def _select_model_dataset(self):
        selected = self.load_data('datasets/select_models_datasets.json')
        self.group = self.cluster_path.split('.csv')[0][-1]  if self.data_name == 'embedllm' else 1 # for group1 or group2 of embedLLM
        
        if 'routerbench' in self.data_path.lower():
            self.selected_models = selected['routerbench']['selected_models']
            self.new_models = selected['routerbench']['new_models'] #selected as OOD models
            self.new_datasets = selected['routerbench']['new_datasets'] #selected as OOD datasets
            self.select_datasets = selected['routerbench']['selected_datasets'] 
            # self.columns = [c for c in data.columns.tolist() if c not in self.new_models]
        elif 'embedllm' in self.data_path.lower():
            self.selected_models = selected[f'embedllm{self.group}']['selected_models']
            self.new_models = selected[f'embedllm{self.group}']['new_models'] #selected as OOD models
            self.new_datasets = selected[f'embedllm{self.group}']['new_datasets'] #selected as OOD datasets
            self.select_datasets = selected[f'embedllm{self.group}']['selected_datasets'] 
            # self.columns = [c for c in data.columns.tolist() if c in self.selected_models] + ['prompt_id', 'category', 'prompt', 'dataset', 'pos', 'neg']
        else:
            pass
        self.model_orders = {m: i for i, m in enumerate(self.selected_models + self.new_models)}
        self.order_models = {str(i): m for i, m in enumerate(self.selected_models + self.new_models)}
        self.perf_dimensions = self.select_datasets + ['avg_perf', 'std_perf', 'avg_cost']
        # print(self.model_orders)
        

    def load_clusters(self):
        
        # scaler = MinMaxScaler()
        self.perf = self.load_data(self.cluster_path)
        self.all_model_clusters = dict(zip(self.perf['models'], self.perf['act_cluster']))
        self.clusters = self.perf['act_cluster'].tolist()
        self.n_clusters = max(self.clusters) + 1

        pc = [p/c for p, c in zip(self.perf.loc[self.perf['models'].isin(self.selected_models), 'avg_perf'].tolist(), 
                                                      self.perf.loc[self.perf['models'].isin(self.selected_models), 'avg_cost'].tolist())] #weight for cluster embedding combination
        max_pc = max(pc)
        self.perf['perf_cost'] = self.perf['avg_perf']/self.perf['avg_cost']
        self.perf['perf_cost'] = self.perf['perf_cost']/max_pc
        
        if 'routerbench' in self.data_path.lower():
            self.perf['avg_cost'] = self.perf['avg_cost'] * 1000
        
        if self.is_ood:
            self.selected_models = self.selected_models + self.new_models
            self.new_models = []
        self.n_models = len(self.selected_models)
        model_clusters = {m: c for m, c in self.all_model_clusters.items() if str(m) in self.selected_models}
        self.cluster_models = {}
        for c in range(self.n_clusters):
            if str(c) not in self.cluster_models:
                self.cluster_models[str(c)] = []
            for m in self.selected_models:
                if model_clusters[m] == c:
                    self.cluster_models[str(c)].append(m)
        self.new_model_clusters = {m: c for m, c in self.all_model_clusters.items() if str(m) in self.new_models}
        self.all_order_clusters = {self.model_orders[m]: c for m, c in self.all_model_clusters.items() if str(m) in self.new_models + self.selected_models}
        
        self.perf_vec = [
                torch.tensor(
                    self.perf.loc[self.perf['models'] == m, self.perf_dimensions].values[0],
                    dtype=torch.float32, device = self.device
                ) for m in self.selected_models + self.new_models
            ]
        self.perf_vec = {self.model_orders[m]: p for m, p in zip(self.selected_models + self.new_models, self.perf_vec)}

        return model_clusters
    
    def get_most_common_min_cluster(self, pos):
        counts = Counter([self.model_clusters[m] for m in pos])
        max_count = max(counts.values())
        return min(k for k, v in counts.items() if v == max_count)
    
    def compute_similarity(self, input1, input2):
        input2 = input2.to(self.device)
        input1 = input1.to(self.device)
        # if input1.size(1) == 1:  # (4, 1, 768)
        #     input1 = input1.expand(-1, input2.size(1), -1) 
        if self.simi == "cos":
            return F.cosine_similarity(input1, input2, dim=-1)
        elif self.simi == 'dot':
            return input1 @ input2.permute(0, 2, 1)
        elif self.simi == 'euclidean':
            # Ensure input1 and input2 are the same shape: (B, T, D) or (B, D)
            # If shape is (B, D), unsqueeze to (B, 1, D) for broadcasting
            if input1.dim() == 2:
                input1 = input1.unsqueeze(1)  # (B, 1, D)
            if input2.dim() == 2:
                input2 = input2.unsqueeze(1)  # (B, 1, D)

            # Compute squared difference
            dist = (input1 - input2) ** 2  # (B, T, D)
            dist = dist.sum(dim=-1)  # sum over embedding dim -> (B, T)
            return dist
        
    def construct_cluster_embedding(self, cluster_pos, cluster_neg):
        
        # _ = self.load_cluster_embedding()
        bs = cluster_pos.size(0)
        cluster_pos = cluster_pos.tolist()
        cluster_neg = cluster_neg.tolist()
        
        pos_cluster_embedding, neg_cluster_embedding = [], []

        for b in range(bs):
            # print("cluster_pos[b][0]:", cluster_pos[b][0])
            pos_cluster_embedding_b = [self.cluster_embedding[str(c)] for c in cluster_pos[b]]
            pos_cluster_embedding.append(torch.stack(pos_cluster_embedding_b))
            neg_cluster_embedding_b = [self.cluster_embedding[str(c)] for c in cluster_neg[b]]
            neg_cluster_embedding.append(torch.stack(neg_cluster_embedding_b))
            
        pos_cluster_embedding = torch.stack(pos_cluster_embedding)#.view(bs, 1, self.dim)
        neg_cluster_embedding = torch.stack(neg_cluster_embedding)
        cluster_embedding = torch.cat([pos_cluster_embedding, neg_cluster_embedding], dim = 1)

        return pos_cluster_embedding, neg_cluster_embedding, cluster_embedding
    
    def construct_model_embedding(self, model_index):
        
        bs = model_index.size(0)
        perf_dim = len(self.perf_dimensions)
        # print(self.order_models)
        
        model_embedding = []
        model_perf_vectors = []
        for b in range(bs):
            
            model_ids = model_index[b].tolist()
            model_embedding_b = [self.model_embeddings[str(int(n))] for n in model_ids]
        
            perf_b = [self.perf_vec[p] for p in model_ids]

            model_embedding.append(torch.stack(model_embedding_b))
            model_perf_vectors.append(torch.stack(perf_b))
            # neg_model_perf_vectors.append(torch.stack(neg_perf_b))
            
        model_embedding = torch.stack(model_embedding)
        # neg_model_embedding = torch.stack(neg_model_embedding)
        model_perf_vectors = torch.stack(model_perf_vectors) * 50
        # neg_model_perf_vectors = torch.stack(neg_model_perf_vectors) * 50
        # print(pos_model_embedding.size(), neg_model_embedding.size(), pos_model_perf_vectors.size(), neg_model_perf_vectors.size())

        return model_embedding.to(self.device), model_perf_vectors.to(self.device) #, neg_model_perf_vectors.to(self.device)
        
        
    def forward(self, **inputs):

        x = self.backbone(**inputs)
        # We used the first token as classifier token.
        hidden_state = x['last_hidden_state'][:, 0, :]
        return hidden_state
    
    def compute_sample_cluster_loss(self, hidden_state, pos_cluster_embed, neg_cluster_embed):
        
        pos_simi = torch.exp(self.compute_similarity(hidden_state.unsqueeze(1), pos_cluster_embed)/self.tau)#.sum(dim = 1)
        neg_simi = torch.exp(self.compute_similarity(hidden_state.unsqueeze(1), neg_cluster_embed)/self.tau)
        # pos_greater_than_all = (pos_simi >= neg_simi).all(dim=1)
        # print(pos_simi.size(), neg_simi.size())
        neg_simi = torch.clamp(neg_simi, max=torch.exp(-self.margin)).sum(dim = 1) #add penalty for negative samples
        pos_simi = pos_simi.sum(dim = 1)
        loss = -torch.log(pos_simi / (pos_simi + neg_simi + 1e-8))
        
        return loss.mean()#, pos_greater_than_all
    
    def compute_cluster_cluster_loss(self, pos_cluster_embed, neg_cluster_embed):
        
        pos_neg_simi_loss = torch.exp(self.compute_similarity(pos_cluster_embed, neg_cluster_embed)/self.tau)
        # print(pos_neg_simi_loss.size())
        pos_neg_simi_loss = pos_neg_simi_loss.mean()
        
        return pos_neg_simi_loss
    
    def compute_sample_model_loss(self, hidden_state,
                               model_embedding, model_index, topk, lastk, pos_len):
        
        pos_len = pos_len.view(-1).tolist()
        # model_index_ = model_index

        simi = torch.exp(self.compute_similarity(hidden_state.unsqueeze(1), model_embedding) / self.tau)  # (B, T)

        Top_k, Last_v = [], []
        loss = []
        for b in range(hidden_state.size(0)):
            if pos_len[b] > 0 and (model_embedding.size(1) - pos_len[b]) > 0:
                loss_ = 0
                topk_v = simi[b, :pos_len[b]]
                lastk_v = simi[b, pos_len[b]:]
                
                lastk_v = lastk_v.sum()
                for l in range(pos_len[b]):
                    loss_ += -torch.log(topk_v[l]/(topk_v[l] + lastk_v + 1e-8))
                loss.append(loss_)
        
        # loss = -torch.log(weighted_topk/(weighted_topk + lastk_v + 1e-8))
        return sum(loss)/len(loss)



    
    def compute_model_perf_loss(self, model_embedding,  model_perf_vectors):

    
        projected_embedding = model_embedding @ self.projector  # (B, T, D2)
        # projected_neg_embedding = neg_model_embedding @ self.projector  # (B, T, D2)
        
        mse_loss = F.mse_loss(projected_embedding, model_perf_vectors) + 1e-4 * torch.norm(self.projector, p = 'nuc') # (B, T)
        # similarity_neg = self.compute_similarity(projected_neg_embedding, neg_model_perf_vectors)  # (B, T)

        
        return mse_loss

    

def load_model(model_path):
    # "microsoft/mdeberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side = 'left', padding = True)
    model = AutoModel.from_pretrained(model_path)
    
    return tokenizer, model

        
def save_model(path, model, config, step, save_model=True, is_main_process=True):
    if not is_main_process:
        return
    cp_dir  = config.checkpoint_dir
    if not os.path.exists(os.path.join(path, cp_dir)):
        os.mkdir(os.path.join(path, cp_dir))

    path1   = os.path.join(path, cp_dir, f"step_{step}.pt")
    path2   = os.path.join(path, f"model_embedding_step_{step}.json")
    if save_model:
        print(f"Saving model to {path1}....")
        checkpoint = {'model': model.state_dict(),
                    'step': step,
                    'config': config}
        torch.save(checkpoint, path1)
    
    model_order = model.order_models
    model_embedding =  {k: v.data.tolist() for k, v in model.model_embeddings.items()}
    cluster_embedding = {k: v.data.tolist() for k, v in model.cluster_embedding.items()}
    projector = model.projector.tolist()
    with open(path2, 'w', encoding='utf-8') as f:
        json.dump({'model_order': model_order, 'model_embedding': model_embedding, 'cluster_embedding': cluster_embedding, 'projector': projector}, f, indent = 4)
    print("Saved.")

    
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

def evaluation(model, config, tokenizer, device, save_path, steps, is_ood, is_newmodels, is_main_process=True):
    if not is_main_process:
        return 0
    print("### Eval start...")
    model.eval()
    evaldata = RouterData(config.eval_path, config.cluster_path, config.max_seq_length, tokenizer, is_ood, is_newmodels)
    evalloader = DataLoader(evaldata, batch_size = config.batch_size, shuffle = False)
    cluster_models = model.cluster_models
    if is_newmodels:
        new_models = evaldata.new_models
        new_model_orders = [model.model_orders[m] for m in new_models]
        perf_vec = [model.perf_vec[str(int(o))] for o in new_model_orders]
        projector_inv = torch.linalg.pinv(model.projector)
        new_model_embeddings = [vec.T @ projector_inv for vec in perf_vec]
        
        all_cluster_models = cluster_models
        all_order_clusters = model.all_order_clusters
        for i, o in enumerate(new_model_orders):
            model.model_embeddings[str(int(o))] = new_model_embeddings[i]
            all_cluster_models[all_order_clusters[str(int(o))]].append(new_models[i])
    # order_models = model.order_models
    model_orders = model.model_orders
    cluster_embedding = torch.stack([model.cluster_embedding[str(i)] for i in range(model.n_clusters)])
    score = []
    cost = []
    selected_model = []
    selected_cluster = []
    top_score = 0
    with torch.no_grad():
        for idx_, batch in tqdm(enumerate(evalloader), total = len(evalloader), desc = 'Sample'):
            inputs, cluster_pos, cluster_neg, model_index, model_labels, model_costs,_ = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            cluster_pos = cluster_pos.to(device)
            cluster_neg = cluster_neg.to(device)
            model_index = model_index.to(device)
            
            hidden_state = model.forward(**inputs)

            cluster_simi = model.compute_similarity(hidden_state.unsqueeze(1), cluster_embedding)
            
            best_clusters = torch.topk(cluster_simi, k = 1, dim = -1).indices.tolist() #length = batch size
            
            for cc, c in enumerate(best_clusters):
            # for cc, c in enumerate(range(hidden_state.size(0))):
                # print(f"batch {cc}")
                best_cluster_models = [model_orders[m]  for c_ in c for m in cluster_models[str(c_)]]
                # print(best_cluster_models)
                # model_embedding = torch.stack([model.model_embeddings[str(int(i))] for i in model_index[cc,:].tolist()])
                model_embedding = torch.stack([model.model_embeddings[str(int(i))] for i in best_cluster_models])
                # print(model_embedding.size())
                model_simi = model.compute_similarity(hidden_state[cc,:].view(1, -1), model_embedding)
                # print(model_simi.size())
                best_model_idx = torch.argmax(model_simi, dim = -1)
                # print(best_model_idx)
                best_model_order = best_cluster_models[best_model_idx]
            
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
                selected_cluster.append(c[0])
                top_score += max(model_labels_)
            
            samples_eval = (idx_ + 1) * config.batch_size
            if samples_eval >= config.eval_size:
                break
    score_ = sum(score)
    cost_ = sum(cost)
    print("### Eval ended")
    print(f"Total {samples_eval} samples, score: {score_}/{len(evaldata)} = {round(score_/len(evaldata),4)}, cost: {cost_}\n")
    with open(f"{save_path}/evaluation-step-{steps}-newmodels-{is_newmodels}.json", 'w', encoding='utf-8') as f:
        json.dump({'final_score': score_, 'final_cost': cost_, 'top_score': len(evaldata), 'scores': score, 'costs': cost, 'select_m': selected_model, 'select_c': selected_cluster}, f, indent = 4)
    print("Saved.")
    
    return score_/len(evaldata)
            
    

def train(config):
        # 初始化分布式训练
        
    import os
    print("Environment Variables:", os.environ)
    local_rank = int(os.environ["LOCAL_RANK"])
    print(local_rank)
    if local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        is_main_process = (local_rank == 0)
        print(is_main_process)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main_process = True
    
    # 只在主进程创建保存目录和配置文件
    if is_main_process:
        timestamp = time.time()
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        local_save_path = os.path.join(config.save_folder, local_time)
        os.makedirs(local_save_path, exist_ok=True)
        
        with open(os.path.join(local_save_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(config), f, indent=4)
    else:
        local_save_path = None
    
    setup_seed(config.seed + (config.local_rank if config.local_rank != -1 else 0))
    tokenizer, model = load_model(config.model_path)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    routerdata = RouterData(config.data_path, config.cluster_path, config.max_seq_length, tokenizer, config.is_ood, config.is_newmodels)
    # routerloader1 = DataLoader(routerdata, batch_size = config.batch_size, shuffle = True, worker_init_fn= seed_worker)
    if config.local_rank != -1:
        sampler = DistributedSampler(routerdata, shuffle=True)
    else:
        sampler = None
    
    routerloader = DataLoader(
        routerdata,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    
    routermodel = RouterModule(model, config, device)
    routermodel.to(device)
    
            # 使用DDP包装模型
    if config.local_rank != -1:
        routermodel = DDP(routermodel, device_ids=[config.local_rank], output_device=config.local_rank)
    
    paras_num = count_parameters(model)
    if is_main_process:
        print(f"Total params: {paras_num}")
    

    # 第二类参数：projector 参数，设置较高的学习率
    projector_params = [param for name, param in routermodel.named_parameters() if 'projector' in name]

    # 使用 param_group 来指定不同的学习率
    optimizer = torch.optim.AdamW([
        # {'params': embedding_params, 'lr': config.embedding_lr},  # embedding 参数使用较低学习率
        {'params': projector_params, 'lr': config.lr * 1000},  # projector 参数使用较高学习率
        {'params': [param for param in routermodel.parameters() if param not in projector_params], 'lr': config.lr}  # 其他参数使用默认学习率
    ], lr=config.lr)
    
    
    if config.use_scheduler:
        scheduler = get_scheduler(optimizer, config, len(routerloader))
        # scheduler2 = get_scheduler(optimizer2, config, len(routerloader))
    
    print("Training start...")
    sample_cluster_loss, cluster_cluster_loss, sample_model_loss, model_perf_loss = [], [], [], []
    score = 0
    step1 = 0
    for epoch in tqdm(range(config.epochs)):
        
        if sampler is not None:
            sampler.set_epoch(epoch)
        #stage 1
        print(f"======= Stage 1...")
        for batch1 in routerloader:
            routermodel.train()
            optimizer.zero_grad()
            
            inputs1, cluster_pos, cluster_neg, model_index, _, _, pos_len = batch1
            inputs1 = {k: v.to(device) for k, v in inputs1.items()}

            cluster_pos = cluster_pos.to(device)
            cluster_neg = cluster_neg.to(device)
            model_index = model_index.to(device)
            pos_len = pos_len.to(device)
            # print(cluster_pos)

            hidden_state1 = routermodel.forward(**inputs1)
            # hidden_state1 = hidden_state1.unsqueeze(1)
            # print(hidden_state1.size())
            pos_cluster_embedding, neg_cluster_embedding, cluster_embedding = routermodel.construct_cluster_embedding(cluster_pos, cluster_neg)
            # print(pos_cluster_embedding.size(), neg_cluster_embedding.size())       
            
            
            sample_cluster_loss_  = routermodel.compute_sample_cluster_loss(hidden_state1, pos_cluster_embedding, neg_cluster_embedding)
            # cluster_cluster_loss_ = routermodel.compute_cluster_cluster_loss(pos_cluster_embedding, neg_cluster_embedding)
            
            model_embedding, model_perf_vectors = routermodel.construct_model_embedding(model_index)

            sample_model_loss_ = routermodel.compute_sample_model_loss(hidden_state1, model_embedding,model_index, topk = 3, lastk = 3, pos_len = pos_len)
            model_perf_loss_ = routermodel.compute_model_perf_loss(model_embedding, model_perf_vectors)
            
            sample_cluster_loss.append(sample_cluster_loss_)
            sample_model_loss.append(sample_model_loss_)
            # cluster_cluster_loss.append(cluster_cluster_loss_)
            
            loss1 =   sample_cluster_loss_ + sample_model_loss_ * 1.2 + model_perf_loss_ * config.beta
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(routermodel.parameters(), max_norm=1.0)

            step1 += 1
            if (step1 + 1) % config.gradient_accumulation == 0:
                optimizer.step()
            
            if config.use_scheduler:
                scheduler.step()
                
            
            if is_main_process and (step1 + 1) % config.log_steps == 0:
                print(f"##Epoch {epoch} Stage 1 ## Step {step1 + 1}, projector-lr: {round(optimizer.param_groups[0]['lr'],5)}, other-lr: {round(optimizer.param_groups[1]['lr'],5)}, sample-model loss: {round(sample_model_loss_.item(),5)}, model-perf loss: {round(model_perf_loss_.item(),5)}, sample-cluster loss: {round(sample_cluster_loss_.item(),5)}")#, cluster-cluster loss: {cluster_cluster_loss_.item()}") #, sample-cluster loss: {sample_cluster_loss_.item()}") #, cluster-cluster loss: {cluster_cluster_loss_.item()}")

                    
            if is_main_process and (step1 + 1) % config.eval_steps == 0:
                score_ = evaluation(routermodel, config, tokenizer, device, local_save_path, step1 + 1, config.is_ood,  config.is_newmodels, is_main_process)
                if step1 + 1 >= 200:
                    pass
                    # _ = evaluation(routermodel, config, tokenizer, device, local_save_path, step1 + 1, True,  True)
                    
            if is_main_process and (step1 + 1) % config.save_steps == 0:
                save_model(config.save_folder, model, config, step1 + 1, True, is_main_process)
            
        
    print("\nTraining end!")
    
    training_log = {'sample_cluster_loss': sample_cluster_loss, 'cluster_cluster_loss': cluster_cluster_loss, 'sample_model_loss': sample_model_loss, 'model_perf_loss': model_perf_loss}
    with open(local_save_path + '/training_log.json', 'w', encoding = 'utf-8') as f:
        json.dump(training_log, f, indent = 4)

    if config.local_rank != -1:
        dist.destroy_process_group()

   
def main():
    config = parse_config()
    os.makedirs(config.save_folder, exist_ok=True)
    
    train(config)
    

if __name__ == "__main__":
    
    
    
    #embedllm LR=8e-3, projector-lr = *50, beta = 0.01, test-acc = 1187/2480 = 0.47
    
    LR=5e-4
    MODEL="/sds_wangby/models/zwx/mdeberta-v3-base"
    class config:
        model_path = MODEL
        data_path = 'datasets/RouterBench/routerbench_5shot-train.csv'
        eval_path = 'datasets/RouterBench/routerbench_5shot-test.csv'
        embedding_path = 'none'
        cluster_path = 'datasets/RouterBench/cluster_2_cost_1.2.csv'
        max_seq_length = 512
        dim = 768
        is_ood = False
        is_newmodels = False
        simi = 'cos'
        tau = 0.5
        margin = 0.3
        alpha = 0.5
        beta = 0.0001
        batch_size = 16
        eval_size = 15000
        epochs = 2
        warmup_rate = 0.1
        lr = LR
        gradient_accumulation = 4
        use_scheduler = True
        scheduler = 'cosine'
        c_steps = 200
        eval_steps = 50
        seed = 66
        log_steps = 5
        save_steps = 1000
        save_folder = 'output'
        checkpoint_dir = 'ck_dir'
    main()
            

