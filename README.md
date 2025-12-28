# FANG

## Introduction

This repository contains the official implementation of the AAAI 2026 paper:

**Improving Generalization in LLM Structured Pruning via Function-Aware Neuron Grouping**

## Pipeline Execution

### Step 1 : Context Clustering

```bash
CUDA_VISIBLE_DEVICES=0 python hidden_feature_clustering_wikitext.py \
    --model your_model \
    --nsamples 128 \
    --save_path clusterresults \
    --mlp_cluster \
    --num_experts_mlp 7 \
    --n_jobs 16 \
    --max_iter 25 \
    --pca_components 64
```

### Step 2 : Compute Cluster–Neuron Score Matrix

```bash
CUDA_VISIBLE_DEVICES=0 python split_gradient_get_grads_wikitext.py \
    --model your_model \
    --nsamples 128 \
    --save_path clusterresults \
    --mlp_cluster \
    --pca_components 64 \
    --gate_weights_file clusterresults/gate_weights.pt
```

### Step 3 : Group Neurons

```bash
CUDA_VISIBLE_DEVICES=0 python split_gradient_residual.py \
    --model_path your_model \
    --save_path clusterresults \
    --score_file clusterresults/importance_scores.pt \
    --mlp_cluster \
    --num_experts_mlp 7 \
    --num_experts_residual_mlp 1
```

### Step 4 : Sparsity Allocation

```bash
CUDA_VISIBLE_DEVICES=0 python FC_sparsity_allocation.py \
    --model your_model \
    --nsamples 128 \
    --pruning_ratio 0.4 \
    --Lambda 0.2 \ # 0.5 * pruning_ratio
    --save_dir sparsity_allocation
```

### Step 5 : Prune the Model

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model your_model \
    --nsamples 128 \
    --pruning_ratio 0.4 \
    --tau 9 \
    --prune_method noiter_group_weight \
    --group_method group_res \
    --eval \
    --eval_zero_shot \
    --neuron_indices_file clusterresults/neuron_residual_indices.pt \
    --cluster_center_file clusterresults/gate_weights.pt \
    --sparsity_allocation_file sparsity_allocation/FC_sp40_lambda20.pt
```

## TODO

- [ ] Improve and extend the README documentation
- [ ] Provide clustering results for different models

## Acknowledgements

https://github.com/CASIA-LMC-Lab/FLAP

https://github.com/llmsresearch/darwinlm

https://github.com/OpenSparseLLMs/LLaMA-MoE-v2
