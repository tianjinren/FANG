import argparse
import os 
import numpy as np
import random
import time
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import seed_worker
from typing import Dict, Optional
import types
from accelerate import Accelerator
from accelerate.utils import release_memory
from torch.optim import SGD
from tqdm import tqdm
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP

from importlib.metadata import version
from tqdm import tqdm

from lib.data import get_wikitext2

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model, cache_dir="llm_weights"):
    model = LlamaForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
        
    model.seqlen = 2048
    return model

def forward_llama_mlp_with_backward_hook_bug_fix(self, x):
    # fmt: off
    batch_size, seq_len, hidden_size = x.shape
    x = x.reshape(batch_size * seq_len, hidden_size)  # ---- reshape -----

    gate_proj_output = self.act_fn(self.gate_proj(x))
    up_proj_output = self.up_proj(x)
    gate_up_mm_output = gate_proj_output * up_proj_output
    down_proj_output = self.down_proj(gate_up_mm_output)

    down_proj_output = down_proj_output.reshape(batch_size, seq_len, hidden_size)  # ---- reshape -----
    return down_proj_output
    # fmt: on

def preprocess(
    input_data: torch.Tensor,
) -> Dict:
    
    input_ids = input_data[0]
    targets = input_data[1]
    attention_masks = torch.tensor(
        [[0 if input_id_seq == -100 else 1 for input_id_seq in input_ids[0]]]
    )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_masks,
    )

class CustomDataset(Dataset):
    def __init__(
        self,
        datalist: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.data = datalist

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        ins = self.data[index]
        processed = preprocess(ins)
        ins = {}
        for key in processed:
            ins[key] = processed[key][0]
        return ins

    def state_dict(self):
        return {
            "datapath": self.datapath,
            "seed": self.seed,
            "rng": self.rng.getstate(),
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--save_path', type=str, default=None, help='Directory to save the neuron indices.')
    parser.add_argument('--mlp_cluster', action="store_true", help='Use mlp cluster.')
    parser.add_argument('--attn_cluster', action="store_true", help='Use attn cluster.')
    parser.add_argument('--pca_components', type=int, default=None, help='Number of dim after pca')
    parser.add_argument('--gate_weights_file', type=str, default=None, help='Path to the gate weights file.')
    args = parser.parse_args()
    
    # Setting seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, None)
    device = torch.device("cuda:0")
    model.train()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    start = time.time()

    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    trainenc, testenc = get_wikitext2(args.nsamples, args.seed, 2048, tokenizer)
    train_dataset = CustomDataset(trainenc)
    # 🔍 prepare data loader
    dataloader_params = {
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": True,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = RandomSampler(train_dataset)
        dataloader_params["drop_last"] = True
        dataloader_params["worker_init_fn"] = seed_worker

    dataloader = DataLoader(train_dataset, **dataloader_params)
    optimizer = SGD(model.parameters())

    # validity check
    if not isinstance(model, LlamaForCausalLM):
        raise ValueError("For now the only supported model is LLaMA!")

    # replace forward func (IMPORTANT)
    for layer_id, layer in enumerate(model.model.layers):
        layer.mlp.forward = types.MethodType(
            forward_llama_mlp_with_backward_hook_bug_fix, layer.mlp
        )  # locate block by the name template

    accelerator = Accelerator()
    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)
    device = accelerator.device
    layer_num = accelerator.unwrap_model(model).config.num_hidden_layers
    neuron_num = accelerator.unwrap_model(model).config.intermediate_size
    head_num = accelerator.unwrap_model(model).config.num_attention_heads
    head_size = accelerator.unwrap_model(model).config.hidden_size // head_num
    # hf_device_map = accelerator.unwrap_model(model).hf_device_map

    # 🔍 load gate weights
    gate_weights = torch.load(args.gate_weights_file)

    # 🔍 initialize temp vars
    cached_attention_masks = None
    if args.attn_cluster:
        classified_cluster_ids_attn = {}
        cached_features_intermediate_attn = {}
        importance_scores_attn = {
            i: {
                j: torch.zeros((head_num,), device=device)
                for j in range(len(gate_weights['attn'][i]))  # cluster id
            }
            for i in range(layer_num)  # layer id
        }
        token_count_attn = {
            i: {
                j: 0
                for j in range(len(gate_weights['attn'][i]))  # cluster id
            }
            for i in range(layer_num)  # layer id
        }  # number of classified tokens in each cluster

        def _forward_hook_input_token_attn(module, input, output):
            """This hook captures the input features to the MLP layers and calculates the classified cluster ids for tokens"""
            hidden_states = input[0]
            dev = hidden_states.device
            if len(hidden_states.shape) == 3:
                if hidden_states.shape[0] != 1:
                    raise ValueError("batch size should be 1")
                hidden_states = hidden_states[0]
            if cached_attention_masks is not None:
                hidden_states = hidden_states[cached_attention_masks.to(dev)]

            layer_id = module.layer_id
            if args.pca_components == None:

                feature_distances = torch.cdist(hidden_states, gate_weights['attn'][layer_id].to(hidden_states.dtype).to(hidden_states.device), p=2)
                classified_cluster_ids_attn[layer_id] = torch.argmin(feature_distances, dim=-1)
            else:
                hidden_states_pca = (hidden_states.detach() - gate_weights['attn_mean'][layer_id].to(hidden_states.dtype).to(hidden_states.device)) @  gate_weights['attn_proj'][layer_id].to(hidden_states.dtype).to(hidden_states.device)

                feature_distances = torch.cdist(hidden_states_pca, gate_weights['attn'][layer_id].to(hidden_states.dtype).to(hidden_states.device), p=2)
                classified_cluster_ids_attn[layer_id] = torch.argmin(feature_distances, dim=-1)

        def _forward_hook_intermediate_attn(module, input, output):
            """This hook captures the intermediate features of the neurons"""
            hidden_states = input[0]
            dev = hidden_states.device
            if len(hidden_states.shape) == 3:
                if hidden_states.shape[0] != 1:
                    raise ValueError("batch size should be 1")
                hidden_states = hidden_states[0]
            if cached_attention_masks is not None:
                hidden_states = hidden_states[cached_attention_masks.to(dev)]

            layer_id = module.layer_id
            cached_features_intermediate_attn[layer_id] = hidden_states.detach()

        def _backward_hook_intermediate_attn(module, grad_in, grad_out):
            """This hook captures the backward gradients of the intermediate neurons, and calculates the corresponding importance scores"""
            hidden_states_grad = grad_in[0]
            dev = hidden_states_grad.device
            if len(hidden_states_grad.shape) == 3:
                if hidden_states_grad.shape[0] != 1:
                    raise ValueError("batch size should be 1")
                hidden_states_grad = hidden_states_grad[0]
            if cached_attention_masks is not None:
                hidden_states_grad = hidden_states_grad[cached_attention_masks.to(dev)]

            layer_id = module.layer_id

            # add to the score cache
            for cluster_id in range(gate_weights['attn'][layer_id].shape[0]):  # iterate over clusters
                feature_mask: torch.BoolTensor = (classified_cluster_ids_attn[layer_id] == cluster_id)
                importance_score = hidden_states_grad[feature_mask].detach() * cached_features_intermediate_attn[layer_id][feature_mask]
                importance_score = importance_score.view(importance_score.shape[0], head_num, head_size)

                importance_scores_attn[layer_id][cluster_id] += torch.sum(torch.abs(importance_score), dim=(0,2))
                token_count_attn[layer_id][cluster_id] += feature_mask.sum().item()
            
    if args.mlp_cluster:
        classified_cluster_ids_mlp = {}
        cached_features_intermediate_mlp = {}
        importance_scores_mlp = {
            i: {
                j: torch.zeros((neuron_num,), device=device)
                for j in range(len(gate_weights['mlp'][i]))  # cluster id
            }
            for i in range(layer_num)  # layer id
        }
        token_count_mlp = {
            i: {
                j: 0
                for j in range(len(gate_weights['mlp'][i]))  # cluster id
            }
            for i in range(layer_num)  # layer id
        }

        # 🔍 hooks
        def _forward_hook_input_token_mlp(module, input, output):
            """This hook captures the input features to the MLP layers and calculates the classified cluster ids for tokens"""
            hidden_states = input[0]
            dev = hidden_states.device
            if len(hidden_states.shape) == 3:
                if hidden_states.shape[0] != 1:
                    raise ValueError("batch size should be 1")
                hidden_states = hidden_states[0]
            if cached_attention_masks is not None:
                hidden_states = hidden_states[cached_attention_masks.to(dev)]

            layer_id = module.layer_id
            if args.pca_components == None:

                feature_distances = torch.cdist(hidden_states, gate_weights['mlp'][layer_id].to(hidden_states.dtype).to(hidden_states.device), p=2)
                classified_cluster_ids_mlp[layer_id] = torch.argmin(feature_distances, dim=-1)
            else:
                hidden_states_pca = (hidden_states.detach() - gate_weights['mlp_mean'][layer_id].to(hidden_states.dtype).to(hidden_states.device)) @  gate_weights['mlp_proj'][layer_id].to(hidden_states.dtype).to(hidden_states.device)

                feature_distances = torch.cdist(hidden_states_pca, gate_weights['mlp'][layer_id].to(hidden_states.dtype).to(hidden_states.device), p=2)
                classified_cluster_ids_mlp[layer_id] = torch.argmin(feature_distances, dim=-1)

        def _forward_hook_intermediate_mlp(module, input, output):
            """This hook captures the intermediate features of the neurons"""
            hidden_states = input[0]
            dev = hidden_states.device
            if len(hidden_states.shape) == 3:
                if hidden_states.shape[0] != 1:
                    raise ValueError("batch size should be 1")
                hidden_states = hidden_states[0]
            if cached_attention_masks is not None:
                hidden_states = hidden_states[cached_attention_masks.to(dev)]

            layer_id = module.layer_id
            cached_features_intermediate_mlp[layer_id] = hidden_states.detach()

        def _backward_hook_intermediate_mlp(module, grad_in, grad_out):
            """This hook captures the backward gradients of the intermediate neurons, and calculates the corresponding importance scores"""
            hidden_states_grad = grad_in[0]
            dev = hidden_states_grad.device
            if len(hidden_states_grad.shape) == 3:
                if hidden_states_grad.shape[0] != 1:
                    raise ValueError("batch size should be 1")
                hidden_states_grad = hidden_states_grad[0]
            if cached_attention_masks is not None:
                hidden_states_grad = hidden_states_grad[cached_attention_masks.to(dev)]

            layer_id = module.layer_id

            # add to the score cache
            for cluster_id in range(gate_weights['mlp'][layer_id].shape[0]):  # iterate over clusters
                feature_mask: torch.BoolTensor = (classified_cluster_ids_mlp[layer_id] == cluster_id)
                importance_score = hidden_states_grad[feature_mask].detach() * cached_features_intermediate_mlp[layer_id][feature_mask]

                importance_scores_mlp[layer_id][cluster_id] += torch.sum(torch.abs(importance_score), dim=0)
                token_count_mlp[layer_id][cluster_id] += feature_mask.sum().item()

    # 🔍 start calculating importance scores
    ## initialization
    for layer_id, layer in enumerate(accelerator.unwrap_model(model).model.layers):  # locate block by the name template
        assert isinstance(layer.mlp, LlamaMLP)

        if args.mlp_cluster:

            layer.mlp.up_proj.layer_id = layer_id
            layer.mlp.up_proj.register_forward_hook(_forward_hook_input_token_mlp)

            layer.mlp.down_proj.layer_id = layer_id
            layer.mlp.down_proj.register_forward_hook(_forward_hook_intermediate_mlp)  # input of "down_proj" <==> "up_proj * gate_proj" output
            layer.mlp.down_proj.register_backward_hook(_backward_hook_intermediate_mlp)  # grad_in of "down_proj" <==> grad of "up_proj * gate_proj" output
        
        if args.attn_cluster:
            layer.self_attn.q_proj.layer_id = layer_id
            layer.self_attn.q_proj.register_forward_hook(_forward_hook_input_token_attn)

            layer.self_attn.o_proj.layer_id = layer_id
            layer.self_attn.o_proj.register_forward_hook(_forward_hook_intermediate_attn)
            layer.self_attn.o_proj.register_backward_hook(_backward_hook_intermediate_attn)

    ## forward
    for i, batch in tqdm(enumerate(dataloader)):
        release_memory()

        if accelerator.is_main_process:
            with torch.cuda.device(device):
                print(f"Used GPU memory ({device}) (before forward): " + str(int(torch.cuda.memory_allocated() / 1024 / 1024)) + " MB")

        attention_mask = batch.get("attention_mask")
        cached_attention_masks = attention_mask.bool().flatten() if attention_mask is not None else None
        outputs = model(**batch)

        if accelerator.is_main_process:
            with torch.cuda.device(device):
                print(f"Used GPU memory ({device}) (after forward): " + str(int(torch.cuda.memory_allocated() / 1024 / 1024)) + " MB")

        loss = outputs.loss
        loss.backward()
        optimizer.zero_grad()

    # 🔍 aggregate results
    final_importance_scores = {}

    if args.attn_cluster:
        final_importance_scores['attn'] = {}
    if args.mlp_cluster:
        final_importance_scores['mlp'] = {}

    for layer_id in tqdm(range(layer_num)):
        if args.mlp_cluster:
            this_layer_importance_scores_mlp = {}
            for cluster_id in range(gate_weights['mlp'][layer_id].shape[0]):  # iterate over clusters
                # gather results on different devices
                gathered_this_token_count_mlp = accelerator.reduce(torch.tensor(token_count_mlp[layer_id][cluster_id], device=device), reduction="sum")
                gathered_importance_scores_mlp = accelerator.reduce(importance_scores_mlp[layer_id][cluster_id], reduction="sum")

                # get mean values
                if gathered_this_token_count_mlp > 0:
                    gathered_importance_scores_mlp /= gathered_this_token_count_mlp
                this_layer_importance_scores_mlp[cluster_id] = gathered_importance_scores_mlp.cpu()

                accelerator.print(f"layer {layer_id} mlp cluster {cluster_id}: {gathered_this_token_count_mlp} tokens, {gathered_importance_scores_mlp[:10]}")
            final_importance_scores['mlp'][layer_id] = this_layer_importance_scores_mlp
        
        if args.attn_cluster:
            this_layer_importance_scores_attn = {}
            for cluster_id in range(gate_weights['attn'][layer_id].shape[0]):
                gathered_this_token_count_attn = accelerator.reduce(torch.tensor(token_count_attn[layer_id][cluster_id], device=device), reduction="sum")
                gathered_importance_scores_attn = accelerator.reduce(importance_scores_attn[layer_id][cluster_id], reduction="sum")
                if gathered_this_token_count_attn > 0:
                    gathered_importance_scores_attn /= gathered_this_token_count_attn
                this_layer_importance_scores_attn[cluster_id] = gathered_importance_scores_attn.cpu()
                accelerator.print(f"layer {layer_id} attn cluster {cluster_id}: {gathered_this_token_count_attn} tokens, {gathered_importance_scores_attn[:10]}")
            final_importance_scores['attn'][layer_id] = this_layer_importance_scores_attn

    # 🔍 save to disk
    if accelerator.is_main_process:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        up_filename = os.path.join(args.save_path, f"importance_scores.pt")
        torch.save(final_importance_scores, up_filename)
    accelerator.wait_for_everyone()

    accelerator.print("Done!")
    end = time.time()
    elapsed_minutes = (end - start) / 60
    print(f"allocation: {elapsed_minutes:.2f} minutes")
    # fmt: on

if __name__ == '__main__':
    main()