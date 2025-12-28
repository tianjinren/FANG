import argparse
import os
import pdb
import torch
from tqdm import tqdm
import numpy as np
from transformers import LlamaConfig

from lib.expert_split_residual import GradientSplitResidual
from lib.expert_split import GradientSplit

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def str2bool(v, extended=True):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true",) + (("yes", "t", "y", "1") if extended else ()):
        return True
    elif v.lower() in ("false",) + (("no", "f", "n", "0") if extended else ()):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# fmt: off
class GradientSplitResidualV2(GradientSplitResidual):
    # Here we only use the `split` function in `GradientSplit`.
    # Other functions may raise errors as the format of `self.labels` is changed.
    def __init__(self, config, layer, score_list):
        super().__init__(config, None, layer, score_list)

    def split(
        self,
        expert_num_moe,
        expert_num_residual,
        expert_size,
        criterion="min",
        share_neurons=False,
    ):
        super().split(
            expert_num_moe, expert_num_residual, expert_size, criterion, share_neurons
        )

        new_labels = {}

        new_labels["residual"] = [
            i for neuron_ids in self.labels[:expert_num_residual] for i in neuron_ids
        ]
        for expert_id in range(expert_num_moe):
            new_labels[expert_id] = self.labels[expert_num_residual + expert_id]

        self.labels = new_labels
        print(self.labels)

class GradientSplitV2(GradientSplit):
    # Here we only use the `split` function in `GradientSplit`.
    # Other functions may raise errors as the format of `self.labels` is changed.
    def __init__(self, config, layer, score_list):
        super().__init__(config, None, layer, score_list)

    def split_without_neuron_sharing(self, expert_num, expert_size, criterion):
        super().split_without_neuron_sharing(expert_num, expert_size, criterion)

        self.labels = [
            np.nonzero(self.labels == expert_id)[0].tolist()
            for expert_id in range(expert_num)
        ]
        print(self.labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--score_file', type=str, default=None)
    parser.add_argument('--visualization_path', type=str, default=None)
    parser.add_argument('--mlp_cluster', action="store_true", help='Use mlp cluster.')
    parser.add_argument('--attn_cluster', action="store_true", help='Use attn cluster.')
    parser.add_argument('--expert_size_mlp', type=int, default=None)
    parser.add_argument('--num_experts_mlp', type=int, default=7)
    parser.add_argument('--num_experts_residual_mlp', type=int, default=1)
    parser.add_argument('--expert_size_attn', type=int, default=None)
    parser.add_argument('--num_experts_attn', type=int, default=7)
    parser.add_argument('--num_experts_residual_attn', type=int, default=1)

    parser.add_argument('--criterion', type=str, default="max", choices=("min", "max"))
    parser.add_argument('--share_neurons', type=str, default="False")

    args = parser.parse_args()
    args.share_neurons = str2bool(args.share_neurons)
    print(args, "\n")

    print("Loading llama config...")
    config = LlamaConfig.from_pretrained(args.model_path)

    print("Loading importance scores...")
    all_importance_scores = torch.load(args.score_file)

    # START
    neuron_indices = {}
    if args.mlp_cluster:
        neuron_indices["mlp"] = {}
    if args.attn_cluster:
        neuron_indices["attn"] = {}

    if args.mlp_cluster and args.num_experts_residual_mlp > 0:
        for i in tqdm(range(config.num_hidden_layers)):
            # get scores
            this_layer_scores = all_importance_scores["mlp"][i]
            score_list = [this_layer_scores[j] for j in range(len(this_layer_scores))]

            # check configs
            assert args.num_experts_mlp == len(score_list)

            if args.expert_size_mlp is None:
                args.expert_size_mlp = score_list[0].numel() // (args.num_experts_mlp + args.num_experts_residual_mlp)

            # start split
            split = GradientSplitResidualV2(args, i, score_list)
            split.split(args.num_experts_mlp, args.num_experts_residual_mlp, args.expert_size_mlp, criterion=args.criterion, share_neurons=args.share_neurons)
            neuron_indices["mlp"][i] = split.labels
    if args.mlp_cluster and args.num_experts_residual_mlp == 0:
        for i in tqdm(range(config.num_hidden_layers)):
            # get scores
            this_layer_scores = all_importance_scores['mlp'][i]
            score_list = [this_layer_scores[j] for j in range(len(this_layer_scores))]

            # update configs
            args.num_experts_mlp = len(score_list)

            if args.expert_size_mlp is None:
                args.expert_size_mlp = score_list[0].numel() // args.num_experts_mlp

            # start split
            split = GradientSplitV2(args, i, score_list)
            split.split(args.num_experts_mlp, args.expert_size_mlp, criterion=args.criterion, share_neurons=args.share_neurons)
            neuron_indices['mlp'][i] = split.labels
            args.expert_size_mlp = None  # reset expert size for next layer

    if args.attn_cluster and args.num_experts_residual_attn > 0:
        for i in tqdm(range(config.num_hidden_layers)):
            # get scores
            this_layer_scores = all_importance_scores["attn"][i]
            score_list = [this_layer_scores[j] for j in range(len(this_layer_scores))]

            # check configs
            assert args.num_experts_attn == len(score_list)

            if args.expert_size_attn is None:
                args.expert_size_attn = score_list[0].numel() // (args.num_experts_attn + args.num_experts_residual_attn)

            # start split
            split = GradientSplitResidualV2(args, i, score_list)
            split.split(args.num_experts_attn, args.num_experts_residual_attn, args.expert_size_attn, criterion=args.criterion, share_neurons=args.share_neurons)
            neuron_indices["attn"][i] = split.labels
    if args.attn_cluster and args.num_experts_residual_attn == 0:
        for i in tqdm(range(config.num_hidden_layers)):
            # get scores
            this_layer_scores = all_importance_scores['attn'][i]
            score_list = [this_layer_scores[j] for j in range(len(this_layer_scores))]

            # update configs
            args.num_experts_mlp = len(score_list)

            if args.expert_size_attn is None:
                args.expert_size_attn = score_list[0].numel() // args.num_experts_mlp

            # start split
            split = GradientSplitV2(args, i, score_list)
            split.split(args.num_experts_mlp, args.expert_size_attn, criterion=args.criterion, share_neurons=args.share_neurons)
            neuron_indices['attn'][i] = split.labels
            args.expert_size_attn = None

    # SAVE
    create_dir(args.save_path)
    torch.save(neuron_indices, os.path.join(args.save_path, "neuron_residual_indices.pt"))
    print("Done.")

# fmt: on
