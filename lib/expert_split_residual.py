import itertools
from collections import Counter
import torch
import pdb
import pickle

import io
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

class LayerSplit:
    def __init__(self, config, template, layer):
        self.config = config
        self.template = template
        self.layer = layer

    def save(self):
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)

        filename = os.path.join(self.config.save_path, self.template.format(self.layer))
        torch.save(self.labels, filename, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Expert indices for layer {self.layer} saved to "{filename}".')

    def cnt(self):
        print(Counter(self.labels))

def chunk_list(input_list, num_chunks):

    avg_chunk_size = len(input_list) // num_chunks
    remainder = len(input_list) % num_chunks

    chunks = []
    start = 0
    for _ in range(num_chunks):
        chunk_size = avg_chunk_size + 1 if remainder > 0 else avg_chunk_size
        chunks.append(input_list[start : start + chunk_size])
        start += chunk_size
        remainder -= 1

    return chunks

def visualize_expert_neuron_overlap(
    selected_masks: torch.Tensor,
    num_experts: int,
    intermediate_size: int,
    expert_size: int,
    layer_idx: int,
    save_dir: str = "./",
    save_fig: bool = True,
):
    # fmt: off
    torch.set_printoptions(
        precision=4,  
        threshold=100000,
        edgeitems=3,
        linewidth=160,  
        profile="full",
        sci_mode=False
    )

    """overlap rate between each expert pair"""
    # rate calculation: intersection(Ei, Ej) / union(Ei, Ej)
    intersection_num = torch.mm(selected_masks, selected_masks.transpose(0, 1))
    union_num = torch.full_like(intersection_num, fill_value=intermediate_size) - torch.mm((1 - selected_masks), (1 - selected_masks).transpose(0, 1))
    overlap_rate = intersection_num / union_num

    # print(intersection_num)
    # print(union_num)
    print("overlap_rate", overlap_rate, sep="\n", flush=True)

    """overlap count for each expert"""
    # rows: overlap count,  columns: different experts
    overlap_count = torch.zeros((num_experts, num_experts), dtype=torch.int)

    sum_count = selected_masks.sum(0)  # shape(intermediate_size,)
    selected_masks = selected_masks.bool()
    for overlap_times in range(num_experts):
        this_overlap_neurons = (sum_count == (overlap_times + 1))  # shape(intermediate_size,)
        # print(this_overlap_neurons.sum())
        each_expert_overlap_neurons = selected_masks & this_overlap_neurons  # shape(num_experts, intermediate_size)
        # print(each_expert_overlap_neurons.sum())
        overlap_count[overlap_times, :] = each_expert_overlap_neurons.sum(1)
        # print(overlap_count[overlap_times, :])

    # print(overlap_count.sum(0))
    print("overlap_count", overlap_count, sep="\n", flush=True)

    """save graphs"""
    total_neurons = (sum_count > 0).sum().item()
    overlap_rate = overlap_rate.numpy()
    overlap_count = overlap_count.numpy()

    path_overlap_rate = Path(os.path.join(save_dir, "overlap_rate"))
    if path_overlap_rate.is_file():
        raise ValueError(f"{save_dir} is a file, not a directory")
    path_overlap_rate.mkdir(exist_ok=True, parents=True)

    path_overlap_count = Path(os.path.join(save_dir, "overlap_count"))
    if path_overlap_count.is_file():
        raise ValueError(f"{save_dir} is a file, not a directory")
    path_overlap_count.mkdir(exist_ok=True, parents=True)

    with open(os.path.join(save_dir, "total_neurons.txt"), "a") as file:
        file.write(f"{total_neurons}\n")

    """overlap_rate"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(overlap_rate, vmin=0.0, vmax=1.0, cmap=mpl.colormaps["Greens"], interpolation="nearest", )

    for i in range(overlap_rate.shape[0]):
        for j in range(overlap_rate.shape[1]):
            ax.text(j, i, f"{overlap_rate[i, j]:.4f}", ha="center", va="center", color="black", fontsize=4, )

    ax.set_title(f"Total Selected Neurons {total_neurons} -- Layer {layer_idx}")
    ax.set_axis_off()
    fig.colorbar(im)
    fig.tight_layout()

    """overlap_count"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(overlap_count, vmin=0, vmax=expert_size, cmap=mpl.colormaps["Blues"], interpolation="nearest", )

    for i in range(overlap_count.shape[0]):
        for j in range(overlap_count.shape[1]):
            ax.text(j, i, f"{overlap_count[i, j]}", ha="center", va="center", color="black", fontsize=4, )

    ax.set_title(f"Expert Size {expert_size} -- Layer {layer_idx}")
    ax.set_axis_off()
    fig.colorbar(im)
    fig.tight_layout()

class GradientSplitResidual(LayerSplit):
    # fmt: off
    def __init__(self, config, template, layer, score_list):
        super().__init__(config, template, layer)
        self.score_list = score_list
        self.neuron_num = score_list[0].size(0)

    def sort_by_criterion(self, criterion):
        sorted_score_list = []
        sorted_index_list = []

        for scores in self.score_list:
            if criterion == "min":
                sorted_scores, sorted_indices = scores.sort(0)
            elif criterion == "max":
                sorted_scores, sorted_indices = scores.sort(0, descending=True)
            else:
                raise NotImplementedError

            sorted_score_list.append(sorted_scores.tolist())
            sorted_index_list.append(sorted_indices.tolist())

        return sorted_score_list, sorted_index_list

    def remove_residual_neurons(self, sorted_index_list, residual_neuron_mask):
        new_residual_neuron_mask = []

        for indices in sorted_index_list:
            new_indices = []
            for index in indices:
                if not residual_neuron_mask[index]:
                    new_indices.append(index)
            new_residual_neuron_mask.append(new_indices)

        return new_residual_neuron_mask

    def split_without_neuron_sharing(self, expert_num_moe, expert_num_residual, expert_size, criterion):
        neuron_used_mark = [False] * self.neuron_num
        expert_start_index = [0] * expert_num_moe
        expert_neuron_count = [0] * expert_num_moe
        expert_neuron_count_total = 0

        # select residual neurons with the sharing algorithm
        self.split_with_neuron_sharing(expert_num_moe, expert_num_residual, expert_size, criterion)

        # clear labels for moe neurons
        for expert_id in range(expert_num_residual, expert_num_residual + expert_num_moe):
            self.labels[expert_id] = []

        # exclude neurons selected by the residual block
        residual_labels = self.labels[:expert_num_residual]
        for index in list(itertools.chain(*residual_labels)):
            neuron_used_mark[index] = True

        sorted_score_list, sorted_index_list = self.sort_by_criterion(criterion)

        # iterate over the "sorted_score_list" and compare
        # greedily select the maximum score from the highest score of each expert
        # O(neuron_num * expert_num_moe) time complexity
        moe_neuron_num = expert_num_moe * expert_size
        while expert_neuron_count_total < moe_neuron_num:
            if criterion == "min":
                now_selected_score = float('inf')
            elif criterion == "max":
                now_selected_score = float('-inf')
            else:
                raise NotImplementedError

            now_selected_neuron = -1
            now_selected_expert = -1

            for expert_id in range(expert_num_moe):
                while expert_start_index[expert_id] < self.neuron_num:
                    if neuron_used_mark[sorted_index_list[expert_id][expert_start_index[expert_id]]]:
                        expert_start_index[expert_id] += 1
                    else:
                        break

                if expert_neuron_count[expert_id] == expert_size or expert_start_index[expert_id] == self.neuron_num:
                    continue

                if criterion == "min":
                    if sorted_score_list[expert_id][expert_start_index[expert_id]] <= now_selected_score:  # ----- different here -----
                        now_selected_score = sorted_score_list[expert_id][expert_start_index[expert_id]]
                        now_selected_neuron = sorted_index_list[expert_id][expert_start_index[expert_id]]
                        now_selected_expert = expert_id
                elif criterion == "max":
                    if sorted_score_list[expert_id][expert_start_index[expert_id]] >= now_selected_score:  # ----- different here -----
                        now_selected_score = sorted_score_list[expert_id][expert_start_index[expert_id]]
                        now_selected_neuron = sorted_index_list[expert_id][expert_start_index[expert_id]]
                        now_selected_expert = expert_id
                else:
                    raise NotImplementedError

            self.labels[expert_num_residual + now_selected_expert].append(now_selected_neuron)
            assert (not neuron_used_mark[now_selected_neuron])
            neuron_used_mark[now_selected_neuron] = True
            expert_start_index[now_selected_expert] += 1
            expert_neuron_count[now_selected_expert] += 1
            expert_neuron_count_total += 1
            # print(now_selected_expert, now_selected_neuron)
            # print(expert_neuron_count)
            # print(expert_start_index)

        # print(neuron_used_mark)
        # print(expert_neuron_count)
        # print(expert_start_index)

    def split_with_neuron_sharing(self, expert_num_moe, expert_num_residual, expert_size, criterion="min"):
        sorted_score_list, sorted_index_list = self.sort_by_criterion(criterion)
        residual_labels = []
        residual_neuron_mask = [False] * self.neuron_num 

        # iteratively assign the indices of mostly shared neurons to the residual block
        while not len(residual_labels) >= expert_num_residual * expert_size:
            moe_labels = [sorted_index_list[i][:expert_size] for i in range(expert_num_moe)]

            selected_count = torch.zeros((self.neuron_num,), dtype=torch.int)
            for selected_indices in moe_labels:
                selected_indices = torch.tensor(selected_indices)
                selected_count[selected_indices] += 1

            for repeat_times in range(expert_num_moe, 0, -1):
                repeat_mask = (selected_count == repeat_times)
                selected_neurons_count = torch.sum(repeat_mask).item()
                if selected_neurons_count > 0:
                    residual_indices = torch.nonzero(repeat_mask).flatten().tolist()
                    if len(residual_indices) > expert_num_residual * expert_size - len(residual_labels):  # 如果添加后会超过residual容量上限
                        residual_indices = residual_indices[:expert_num_residual * expert_size - len(residual_labels)]

                    # print(residual_indices)
                    residual_labels.extend(residual_indices)
                    for index in residual_indices:
                        residual_neuron_mask[index] = True
                    print(f"Selected {selected_neurons_count} from repeat {repeat_times}. Total {len(residual_labels)}")
                    break

            sorted_index_list = self.remove_residual_neurons(sorted_index_list, residual_neuron_mask)

        print(f"Final {len(residual_labels)} residual.")
        print(f"Final {self.neuron_num - len(residual_labels)} moe.")

        residual_labels = chunk_list(residual_labels, expert_num_residual) 
        moe_labels = [sorted_index_list[i][:expert_size] for i in range(expert_num_moe)]

        self.labels = residual_labels
        self.labels.extend(moe_labels)

    def split(self, expert_num_moe, expert_num_residual, expert_size, criterion="min", share_neurons=False):
        assert expert_size <= self.neuron_num
        if not share_neurons:
            # print("***", expert_size, expert_num, self.neuron_num)
            if expert_size * (expert_num_moe + expert_num_residual) != self.neuron_num:
                raise ValueError(
                    f'The number of neurons must be exactly divided by the number of experts for non-shared split!\n'
                    f'Now the number of neurons is {self.neuron_num}, but the number of experts is {expert_num_moe} (normal experts) + {expert_num_residual} (residual experts) = {expert_num_moe + expert_num_residual} (total experts), and each expert has {expert_size} neurons.\n'
                    f'The total number of neurons in all experts {(expert_num_moe + expert_num_residual) * expert_size} != {self.neuron_num}.'
                )
            self.split_without_neuron_sharing(expert_num_moe, expert_num_residual, expert_size, criterion)
        else:
            self.split_with_neuron_sharing(expert_num_moe, expert_num_residual, expert_size, criterion)

    def visualize(self, save_path, share_neurons=False):
        if share_neurons:
            num_experts = len(self.labels)
            expert_size = len(self.labels[0])

            selected_mask_list = []
            for i, indices in enumerate(self.labels):
                indices_tensor = torch.tensor(indices)
                selected_mask = torch.zeros((self.neuron_num,), dtype=torch.int)
                selected_mask[indices_tensor] += 1
                selected_mask_list.append(selected_mask)
            selected_masks = torch.stack(selected_mask_list, dim=0)  # shape(num_experts, intermediate_size)

            visualize_expert_neuron_overlap(selected_masks, num_experts, self.neuron_num, expert_size, self.layer, save_dir=save_path)
        else:
            print("Skip visualization as share_neurons==False.")
    # fmt: on
