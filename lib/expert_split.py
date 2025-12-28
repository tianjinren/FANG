import os
import pickle
import random
from collections import Counter

import numpy as np
import torch
import shutil

from lib.expert_split_residual import visualize_expert_neuron_overlap


def delete_file_or_dir(dir):
    if os.path.isfile(dir):
        os.remove(dir)
    elif os.path.exists(dir):
        shutil.rmtree(dir)
    else:
        pass


def pass_kernel_function(tensor, criterion):
    if criterion == "plain":
        return tensor
    elif criterion == "l1_norm":
        return torch.abs(tensor)
    elif criterion == "l2_norm":
        return tensor * tensor
    else:
        raise NotImplementedError


def load_ffn_weight(model, template, layer):
    key = template.format(layer)
    return model[key].numpy()


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


class RandomSplit(LayerSplit):
    def __init__(self, config, model_config, template, layer):
        super().__init__(config, template, layer)
        self.model_config = model_config
        self.neuron_num = model_config.intermediate_size
        self.split_size = self.neuron_num // self.config.num_experts

    def split(self):
        self.labels = np.arange(0, self.config.num_experts, dtype=int).tolist()  # list
        self.labels = self.labels * self.split_size
        random.shuffle(self.labels)


class GradientSplit(LayerSplit):
    # fmt: off
    def __init__(self, config, template, layer, score_list):
        super().__init__(config, template, layer)
        self.score_list = score_list
        self.neuron_num = score_list[0].size(0)
        self.labels = np.zeros((self.neuron_num,))

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

    def split_without_neuron_sharing(self, expert_num, expert_size, criterion):
        sorted_score_list, sorted_index_list = self.sort_by_criterion(criterion)

        # iterate over the "sorted_score_list" and compare
        # greedily select the maximum score from the highest score of each expert
        # O(neuron_num * expert_num) time complexity
        neuron_used_mark = [False] * self.neuron_num
        expert_start_index = [0] * expert_num
        expert_neuron_count = [0] * expert_num
        expert_neuron_count_total = 0

        while expert_neuron_count_total < self.neuron_num:
            if criterion == "min":
                now_selected_score = float('inf')
            elif criterion == "max":
                now_selected_score = float('-inf')
            else:
                raise NotImplementedError

            now_selected_neuron = -1
            now_selected_expert = -1

            for expert_id in range(expert_num):
                if expert_neuron_count[expert_id] == expert_size or expert_start_index[expert_id] == self.neuron_num:
                    continue

                while expert_start_index[expert_id] < self.neuron_num:
                    if neuron_used_mark[sorted_index_list[expert_id][expert_start_index[expert_id]]]:
                        expert_start_index[expert_id] += 1
                    else:
                        break

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

            self.labels[now_selected_neuron] = now_selected_expert
            assert (not neuron_used_mark[now_selected_neuron])
            neuron_used_mark[now_selected_neuron] = True
            expert_start_index[now_selected_expert] += 1
            expert_neuron_count[now_selected_expert] += 1
            expert_neuron_count_total += 1
            # print(now_selected_neuron, now_selected_expert)

        # print(neuron_used_mark)
        # print(expert_neuron_count)
        # print(expert_start_index)

    def split_with_neuron_sharing(self, expert_num, expert_size, criterion):
        sorted_score_list, sorted_index_list = self.sort_by_criterion(criterion)
        self.labels = [sorted_index_list[i][:expert_size] for i in range(expert_num)]

    def split(self, expert_num, expert_size, criterion="min", share_neurons=False):
        assert expert_size <= self.neuron_num
        if not share_neurons:
            # print("***", expert_size, expert_num, self.neuron_num)
            if expert_size * expert_num != self.neuron_num:
                raise ValueError(
                    f'The number of neurons must be exactly divided by the number of experts for non-shared split!\n'
                    f'Now the number of neurons is {self.neuron_num}, but the number of experts is {expert_num} and each expert has {expert_size} neurons.\n'
                    f'The total number of neurons in all experts {expert_num * expert_size} != {self.neuron_num}.'
                )
            self.split_without_neuron_sharing(expert_num, expert_size, criterion)
        else:
            self.split_with_neuron_sharing(expert_num, expert_size, criterion)

    def visualize(self, save_path, share_neurons=True):
        if share_neurons:
            delete_file_or_dir(os.path.join(save_path, "total_neurons.txt"))
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