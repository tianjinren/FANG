import argparse
import os 
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from transformers import AutoTokenizer, AutoModelForCausalLM

from importlib.metadata import version
import pdb
from tqdm import tqdm

from lib.prune import prepare_calibration_input, find_layers
from lib.eval import eval_ppl, eval_zero_shot
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

def sparsity_allocation(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args)
    layers = model.model.layers

    FC_mlp = []
    FC_attn = []

    # Split into sub-problems, separate statistics for each module
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)
        else:
            dev = device

        FCscore_mlp = 0
        FCscore_attn = 0
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                cos_sim_mlp = F.cosine_similarity(inps[j], outs[j], dim=-1)
                mean_cos_sim_mlp = cos_sim_mlp.mean()
                FCscore_mlp += 1 - mean_cos_sim_mlp.item()
                FCscore_attn += 1 - mean_cos_sim_mlp.item()
        FC_mlp.append(FCscore_mlp / args.nsamples)
        FC_attn.append(FCscore_attn / args.nsamples)
        inps, outs = outs, inps 
        torch.cuda.empty_cache()

    allocation = {}
    FC_mlp = np.array(FC_mlp)
    FC_mlp = ((FC_mlp - FC_mlp.min()) * (1 / (FC_mlp.max() - FC_mlp.min()) * args.Lambda * 2))
    all_mlp_ratio = args.pruning_ratio - (FC_mlp - np.mean(FC_mlp))
    allocation['mlp'] = all_mlp_ratio
    FC_attn = np.array(FC_attn)
    FC_attn = ((FC_attn - FC_attn.min()) * (1 / (FC_attn.max() - FC_attn.min()) * args.Lambda * 2))
    all_attn_ratio = args.pruning_ratio - (FC_attn - np.mean(FC_attn)) # make real sparsity equals to args.pruning_ratio
    allocation['attn'] = all_attn_ratio
        
    return allocation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--pruning_ratio', type=float, default=0.4, help='Pruning ratio.')
    parser.add_argument('--Lambda', type=float, default=0.2, help='Lambda for sparsity allocation')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the neuron indices.')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Build the model and tokenizer
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, None)
    device = torch.device("cuda:0")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if "30b" in args.model or "65b" in args.model or "70b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    # cluster the neurons
    allocation = sparsity_allocation(args, model, tokenizer, device)

    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(allocation, args.save_dir + '/FC_sp{}_lambda{}.pt'.format(int(args.pruning_ratio * 100), int(args.Lambda * 100)))

if __name__ == '__main__':
    main()