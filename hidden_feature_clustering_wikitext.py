import argparse
import os 
import numpy as np
import math
import time
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from k_means_constrained import KMeansConstrained
from tqdm import tqdm
from sklearn.decomposition import PCA

from importlib.metadata import version
import pdb
from lib.data import get_loaders

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    model.seqlen = 2048
    return model

def prepare_calibration_input(model, dataloader, device, args):

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids

class Clusters:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.out_dim = layer.weight.data.shape[0]
        self.in_dim = layer.weight.data.shape[1]
        self.nsamples = 0

        self.hidden_states = []

    def add_batch(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))

        self.hidden_states.append(inp)
        self.nsamples += batch_size

    def kmeans(self, args, num_experts=7):
        all_clustering_hidden_states = torch.cat(self.hidden_states, dim=0)
        all_clustering_hidden_states = all_clustering_hidden_states.cpu().numpy()
        num_features = all_clustering_hidden_states.shape[0]
        balanced_num_features = num_features / num_experts
        balance_jitter_factor = max(0.0, args.balance_jitter_factor)
        min_cluster_size = max(0, math.floor(balanced_num_features * (1 - balance_jitter_factor)))
        max_cluster_size = min(num_features, math.ceil(balanced_num_features * (1 + balance_jitter_factor)))

        if args.pca_components == None:
            kmeans = KMeansConstrained(
                n_clusters=num_experts,
                size_min=min_cluster_size,
                size_max=max_cluster_size,
                tol=1e-3,
                n_init=args.n_jobs,
                max_iter=args.max_iter,
                random_state=args.seed,
                n_jobs=args.n_jobs,
                verbose=True,
            ).fit(all_clustering_hidden_states, None)
            gate_weights = torch.from_numpy(kmeans.cluster_centers_)
            del all_clustering_hidden_states
            return gate_weights
        else:
            pca = PCA(n_components=args.pca_components)
            all_clustering_hidden_states_pca = pca.fit_transform(all_clustering_hidden_states)
            projection_matrix = pca.components_.T
            all_clustering_hidden_states_mean = np.mean(all_clustering_hidden_states, axis=0)
            kmeans = KMeansConstrained(
                n_clusters=num_experts,
                size_min=min_cluster_size,
                size_max=max_cluster_size,
                tol=1e-3,
                n_init=args.n_jobs,
                max_iter=args.max_iter,
                random_state=args.seed,
                n_jobs=args.n_jobs,
                verbose=True,
            ).fit(all_clustering_hidden_states_pca, None)
            gate_weights = torch.from_numpy(kmeans.cluster_centers_)
            projection_matrix = torch.from_numpy(projection_matrix)
            all_clustering_hidden_states_mean = torch.from_numpy(all_clustering_hidden_states_mean)
            del all_clustering_hidden_states
            return gate_weights, projection_matrix, all_clustering_hidden_states_mean
        
    def free(self):
        self.hidden_states = [None] * len(self.hidden_states)
        self.nsamples = 0
        torch.cuda.empty_cache()  

def make_kmeans(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print("loading calibdation data")
    dataloader, _ = get_loaders(args.calibration, nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args)
    layers = model.model.layers
    headsize = model.model.config.hidden_size // model.model.config.num_attention_heads
    all_gate_weights = {}
    if args.mlp_cluster:
        all_gate_weights['mlp'] = {}
        if args.pca_components != None:
            all_gate_weights['mlp_mean'] = {}
            all_gate_weights['mlp_proj'] = {}
    if args.attn_cluster:
        all_gate_weights['attn'] = {}
        if args.pca_components != None:
            all_gate_weights['attn_mean'] = {}
            all_gate_weights['attn_proj'] = {}

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        attn = layer.self_attn.q_proj
        mlp = layer.mlp.up_proj

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)

        if args.mlp_cluster:

            wrapped_mlp = Clusters(mlp)          

            def add_batch():
                def tmp(_, inp, out):
                    wrapped_mlp.add_batch(inp[0].data)
                return tmp

            handle = mlp.register_forward_hook(add_batch())
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            handle.remove()

            if args.pca_components == None:
                gate_weights_mlp = wrapped_mlp.kmeans(args, num_experts=args.num_experts_mlp)
                all_gate_weights['mlp'][i] = gate_weights_mlp
            else:
                gate_weights_mlp, proj_mlp, mean_mlp = wrapped_mlp.kmeans(args, num_experts=args.num_experts_mlp)
                all_gate_weights['mlp'][i] = gate_weights_mlp
                all_gate_weights['mlp_mean'][i] = mean_mlp
                all_gate_weights['mlp_proj'][i] = proj_mlp
            wrapped_mlp.free()

        if args.attn_cluster:
            wrapped_attn = Clusters(attn)          

            def add_batch():
                def tmp(_, inp, out):
                    wrapped_attn.add_batch(inp[0].data)
                return tmp

            handle = attn.register_forward_hook(add_batch())
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            handle.remove()

            if args.pca_components == None:
                gate_weights_attn = wrapped_attn.kmeans(args, num_experts=args.num_experts_attn)
                all_gate_weights['attn'][i] = gate_weights_attn
            else:
                gate_weights_attn, proj_attn, mean_attn = wrapped_attn.kmeans(args, num_experts=args.num_experts_attn)
                all_gate_weights['attn'][i] = gate_weights_attn
                all_gate_weights['attn_mean'][i] = mean_attn
                all_gate_weights['attn_proj'][i] = proj_attn
            wrapped_attn.free()

        inps, outs = outs, inps
        torch.cuda.empty_cache()
    return all_gate_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=32, help='Number of calibration samples.')
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument('--save_path', type=str, default='clusterresults/tmp', help='Path to save the cluster centers.')
    parser.add_argument('--mlp_cluster', action="store_true", help='Use mlp cluster.')
    parser.add_argument('--attn_cluster', action="store_true", help='Use attn cluster.')
    parser.add_argument('--num_experts_mlp', type=int, default=7, help='Number of experts.')
    parser.add_argument('--num_experts_attn', type=int, default=7, help='Number of experts.')
    parser.add_argument('--balance_jitter_factor', type=float, default=0.4, help='Jitter factor for balanced clustering.')
    parser.add_argument('--n_jobs', type=int, default=10, help='Number of jobs for KMeans.')
    parser.add_argument('--max_iter', type=int, default=50, help='Max iterations for KMeans.')
    parser.add_argument('--pca_components', type=int, default=None, help='Number of dim after pca')
    parser.add_argument('--pooling', type=int, default=1, help='Pooling method for the clusters.')
    parser.add_argument('--calibration', type=str, default='wikitext2', choices=['c4', 'ptb', 'wikitext2'], help='Calibration dataset.')
    
    args = parser.parse_args()
    
    # Setting seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build the model and tokenizer
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    device = torch.device("cuda:0")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if "30b" in args.model or "65b" in args.model or "70b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    start = time.time()
    all_gate_weights = make_kmeans(args, model, tokenizer, device)
    end = time.time()
    elapsed_minutes = (end - start) / 60
    print(f"cluster: {elapsed_minutes:.2f} minutes")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    torch.save(all_gate_weights, os.path.join(args.save_path, "gate_weights.pt"))
    

if __name__ == '__main__':
    main()