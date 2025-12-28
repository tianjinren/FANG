from typing import List, Union, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.modules.conv import _ConvNd
import numpy as np
import pdb

from lib import dist_utils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def inv_sym(X: Tensor):
    """
    More efficient and stable inversion of symmetric matrices.
    """
    return torch.cholesky_inverse(torch.linalg.cholesky(X))

class FastOBCStruct:

    def __init__(self, layer: nn.Module, rel_damp: float = 1e-2, is_attn: bool = False, verbose: bool = False):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = layer.weight.shape[0], np.prod(layer.weight.shape[1:])
        # FastOBC hyperparameters
        self.rel_damp = rel_damp
        # backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init hessian
        self.H = None
        self.num_samples = 0
        # misc args
        self.verbose = verbose
        self.is_attn = is_attn

    @staticmethod
    def _validate_layer(layer):
        assert isinstance(layer, (nn.Linear, _ConvNd)), "FastOBC supports only linear and convolutional layers."

    # preparatory methods
    @torch.no_grad()
    def update(self, input: Tensor) -> None:
        """
        Update the estimate of Hessian matrix from a batch of data.

        Args:
            input: batch of layer inputs
        """
        # get batch size
        batch_size = input.shape[0]
        # init hessian
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        # input reshaping
        if isinstance(self.layer, nn.Linear):
            input = input.reshape(-1, input.shape[-1])
        else:
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            # output size (batch_size, channels * \prod kernel_size, num_patches)
            input = unfold(input)
            input = input.transpose(1, 2).flatten(0, 1)
        # cast input to float32 before addition
        input = input.float()
        # hessian update
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        self.H.addmm_(input.T, input, beta=beta, alpha=alpha)
        # update number of collected samples
        self.num_samples += batch_size

    def reset(self) -> None:
        self.W = None
        self.H = None
        self.num_samples = 0
        torch.cuda.empty_cache()

    @torch.no_grad()
    def pruning_pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        # 1) Hessian preparation
        assert self.H is not None, "One has to process at least one sample of calibration data to run pruning"
        # synchronize Hessians
        # if dist_utils.is_dist_available_and_initialized():
        #     dist.all_reduce(self.H, op=dist.ReduceOp.AVG)
        # # get ids of pruned channels
        # pruned_ids = torch.diag(self.H) == 0
        # self.H[pruned_ids, pruned_ids] = 1
        # Hessian regularization
        damp = self.rel_damp * torch.diag(self.H).mean()
        self.H[range(self.d_col), range(self.d_col)] += damp
        # 2) Weight preparation
        # copy weight, flatten and convert to float
        self.W = self.W.clone().float()
        if isinstance(self.layer, _ConvNd):
            self.W = self.W.flatten(1, -1)
        # self.W[:, pruned_ids] = 0
        # flag pre step as completed
        # self.pre_step_completed = True

    # mostly copy pasted from ZipLM prune_struct method
    # I assumethis can be significantly optimized by not iterating over every single column to remove (do in blocks)
    def step(self, pruning_ratio: float, headsize: int) -> List[Tensor]:
        d_col, device, dtype = self.d_col, self.W_device, self.W_dtype
        torch.cuda.empty_cache()
        # prepare weight and Cholesky of H^{-1}
        W, Hinv = self._prepare()

        count = d_col // headsize
        num_threshold = int(pruning_ratio * count)
        Losses = torch.zeros(count + 1, device=device, dtype=torch.float)
        mask = torch.zeros(count, device=device).bool()
        rangecount = torch.arange(count, device=device)
        rangecolumns = torch.arange(d_col, device=device)

        if headsize == 1:
            for dropped in range(count + 1):
                diag = torch.diagonal(Hinv)
                scores = torch.sum(W ** 2, 0) / diag
                scores[mask] = float('inf')
                j = torch.argmin(scores)
                Losses[dropped] = scores[j]
                row = Hinv[j, :]
                d = diag[j]
                W -= ((W[:, j] / d).unsqueeze(1)).matmul(row.unsqueeze(0))
                mask[j] = True
                W[:, mask] = 0
                row /= torch.sqrt(d)
                Hinv -= row.unsqueeze(1).matmul(row.unsqueeze(0))
                if dropped + 1 == num_threshold:
                    break
        else:
            mask1 = torch.zeros(d_col, device=device).bool()
            for dropped in range(count + 1):
                blocks = Hinv.reshape(count, headsize, count, headsize)
                blocks = blocks[rangecount, :, rangecount, :]
                try:
                    invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                except:
                    invblocks = torch.linalg.pinv(blocks, hermitian=True)
                W1 = W.reshape((self.d_row, count, headsize)).transpose(0, 1)
                lambd = torch.bmm(W1, invblocks)
                scores = torch.sum(lambd * W1, (1, 2))
                scores[mask] = float('inf')
                j = torch.argmin(scores)
                Losses[dropped] = scores[j]
                rows = Hinv[(headsize * j):(headsize * (j + 1)), :]
                d = invblocks[j]
                W -= lambd[j].matmul(rows)
                mask[j] = True
                mask1[(headsize * j):(headsize * (j + 1))] = True
                W[:, mask1] = 0
                Hinv -= rows.t().matmul(d.matmul(rows))
                Hinv[rangecolumns[mask1], rangecolumns[mask1]] = 1
                if dropped + 1 == num_threshold:
                    break
 
        return W.to(dtype), mask

    def prune_struct(self, pruning_ratio, headsize: int = 1) -> List[Tensor]:
        self.pruning_pre_step()
        sparse_weights, mask= self.step(pruning_ratio, headsize)
        return sparse_weights, mask

    @torch.no_grad()
    def _prepare(self):
        w = self.W
        # get columns with all zeros
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        H = self.H
        # mask rows with zero input channels
        H[zero_cols, :] = 0
        H[:, zero_cols] = 0
        H[zero_cols, zero_cols] = 1
        # invert
        H = inv_sym(H)
        ###H_inv_cho = torch.linalg.cholesky(H, upper=True)
        return w, H #H_inv_cho
    
class FastOBCStruct_group:

    def __init__(self, layer: nn.Module, rel_damp: float = 1e-2, is_attn: bool = False, headsize: int = 1, verbose: bool = False, group_num: int = 7, neuron_indices: Optional[Union[list, dict]] = None, group_method: str = 'group_res'):
        self._validate_layer(layer)
        self.layer = layer
        self.d_row, self.d_col = layer.weight.shape[0], np.prod(layer.weight.shape[1:])
        # FastOBC hyperparameters
        self.rel_damp = rel_damp
        # backup layer properties
        self.W = self.layer.weight
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init hessian
        self.num_samples = 0
        self.verbose = verbose
        self.is_attn = is_attn
        self.group_num = group_num
        self.group_method = group_method
        if is_attn and headsize > 1:
            if isinstance(neuron_indices, dict):
                self.neuron_indices = {}
                for k in neuron_indices.keys():
                    self.neuron_indices[k] = [i * headsize + j for i in neuron_indices[k] for j in range(headsize)]
            if isinstance(neuron_indices, list):
                self.neuron_indices = [None] * len(neuron_indices)
                for k in range(len(neuron_indices)):
                    self.neuron_indices[k] = [i * headsize + j for i in neuron_indices[k] for j in range(headsize)]
        else:
            self.neuron_indices = neuron_indices
        if self.group_method == 'group_res':
            self.Hs = [None] * (group_num - 1)
        elif self.group_method == 'group_nores':
            self.Hs = [None] * group_num    
        elif self.group_method == 'onlyres':
            self.H = None
        else:
            raise ValueError(f"Invalid group method: {self.group_method}")
        

    @staticmethod
    def _validate_layer(layer):
        assert isinstance(layer, (nn.Linear, _ConvNd)), "FastOBC supports only linear and convolutional layers."

    # preparatory methods
    @torch.no_grad()
    def update(self, input: Tensor) -> None:
        """
        Update the estimate of Hessian matrix from a batch of data.

        Args:
            input: batch of layer inputs
        """
        # get batch size
        batch_size = input.shape[0]
        # init hessian
        if self.group_method == 'group_res' or self.group_method == 'group_nores':
            for i in range(len(self.Hs)):
                if self.Hs[i] is None:
                    d_col = len(self.neuron_indices[i])
                    self.Hs[i] = torch.zeros((d_col, d_col), device=input.device, dtype=torch.float32)
        else:
            if self.H is None:
                self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        # input reshaping
        if isinstance(self.layer, nn.Linear):
            input = input.reshape(-1, input.shape[-1])
        else:
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            # output size (batch_size, channels * \prod kernel_size, num_patches)
            input = unfold(input)
            input = input.transpose(1, 2).flatten(0, 1)
        # cast input to float32 before addition
        input = input.float()
        # hessian update
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        if self.group_method == 'group_res' or self.group_method == 'group_nores':
            for k in range(len(self.Hs)):
                neuron_indices = self.neuron_indices[k]
                selected_input = input[:,neuron_indices]
                self.Hs[k].addmm_(selected_input.T, selected_input, beta=beta, alpha=alpha)
        else:
            self.H.addmm_(input.T, input, beta=beta, alpha=alpha)
        self.num_samples += batch_size

    def reset(self) -> None:
        self.W = None
        if self.group_method == 'group_res':
            self.Hs = [None] * (self.group_num - 1)
        elif self.group_method == 'group_nores':
            self.Hs = [None] * self.group_num    
        elif self.group_method == 'onlyres':
            self.H = None
        else:
            raise ValueError(f"Invalid group method: {self.group_method}")
        self.num_samples = 0
        torch.cuda.empty_cache()

    @torch.no_grad()
    def pruning_pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        self.W = self.W.clone().float()
        if isinstance(self.layer, _ConvNd):
            self.W = self.W.flatten(1, -1)
        if self.group_method == 'group_res' or self.group_method == 'group_nores':
            for k in range(len(self.Hs)):
                assert self.Hs[k] is not None, "One has to process at least one sample of calibration data to run pruning"
                damp = self.rel_damp * torch.diag(self.Hs[k]).mean()
                d_col = len(self.neuron_indices[k])
                self.Hs[k][range(d_col), range(d_col)] += damp
        else:
            assert self.H is not None, "One has to process at least one sample of calibration data to run pruning"
            damp = self.rel_damp * torch.diag(self.H).mean()
            self.H[range(self.d_col), range(self.d_col)] += damp
        if self.group_method == 'group_res' or self.group_method == 'onlyres':
            if 'residual' not in self.neuron_indices:
                raise ValueError(f"There is no residual group in neuron_indices!")      

    # mostly copy pasted from ZipLM prune_struct method
    # I assumethis can be significantly optimized by not iterating over every single column to remove (do in blocks)
    def step(self, pruning_ratio: float, headsize: int):
        device, dtype = self.W_device, self.W_dtype
        torch.cuda.empty_cache()
        # prepare weight and Cholesky of H^{-1}
        origW = self.W.clone()
        totalmask = torch.zeros(self.d_col, device=device).bool()
        if self.group_method == 'group_res' or self.group_method == 'onlyres':
            pruning_ratio = pruning_ratio * self.group_num / (self.group_num - 1)
        if self.group_method == 'group_res' or self.group_method == 'group_nores':
            Ws, Hinvs = self._prepare()
            for k in range(len(Hinvs)):
                W, Hinv = Ws[k], Hinvs[k]
                d_col = W.shape[1]

                count = d_col // headsize
                num_threshold = int(pruning_ratio * count)
                mask = torch.zeros(d_col, device=device).bool()
                rangecount = torch.arange(count, device=device)
                rangecolumns = torch.arange(d_col, device=device)
                neuron_indices = self.neuron_indices[k]

                if headsize == 1:
                    for dropped in range(count + 1):
                        diag = torch.diagonal(Hinv)
                        scores = torch.sum(W ** 2, 0) / diag
                        scores[mask] = float('inf')
                        j = torch.argmin(scores)
                        row = Hinv[j, :]
                        d = diag[j]
                        W -= ((W[:, j] / d).unsqueeze(1)).matmul(row.unsqueeze(0))
                        mask[j] = True
                        W[:, mask] = 0
                        row /= torch.sqrt(d)
                        Hinv -= row.unsqueeze(1).matmul(row.unsqueeze(0))
                        if dropped + 1 == num_threshold:
                            break
                else:
                    mask1 = torch.zeros(count, device=device).bool()
                    for dropped in range(count + 1):
                        blocks = Hinv.reshape(count, headsize, count, headsize)
                        blocks = blocks[rangecount, :, rangecount, :]
                        try:
                            invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                        except:
                            invblocks = torch.linalg.pinv(blocks, hermitian=True)
                        W1 = W.reshape((self.d_row, count, headsize)).transpose(0, 1)
                        lambd = torch.bmm(W1, invblocks)
                        scores = torch.sum(lambd * W1, (1, 2))
                        scores[mask1] = float('inf')
                        j = torch.argmin(scores)
                        rows = Hinv[(headsize * j):(headsize * (j + 1)), :]
                        d = invblocks[j]
                        W -= lambd[j].matmul(rows)
                        mask1[j] = True
                        mask[(headsize * j):(headsize * (j + 1))] = True
                        W[:, mask] = 0
                        Hinv -= rows.t().matmul(d.matmul(rows))
                        Hinv[rangecolumns[mask], rangecolumns[mask]] = 1
                        if dropped + 1 == num_threshold:
                            break
                origW[:,neuron_indices] = W
                totalmask[neuron_indices] = mask
        elif self.group_method == 'onlyres':
            w_nores, Hinvs_nores = self._prepare_res() 
            d_col = w_nores.shape[1]
            count = d_col // headsize
            num_threshold = int(pruning_ratio * count)
            mask = torch.zeros(d_col, device=device).bool()
            rangecount = torch.arange(count, device=device)
            rangecolumns = torch.arange(d_col, device=device)

            if headsize == 1:
                res_mask = torch.zeros(self.d_col, device=self.W_device).bool()
                res_mask[self.neuron_indices['residual']] = True
                for dropped in range(count + 1):
                    diag = torch.diagonal(Hinvs_nores)
                    scores = torch.sum(w_nores ** 2, 0) / diag
                    scores[mask] = float('inf')
                    j = torch.argmin(scores)
                    row = Hinvs_nores[j, :]
                    d = diag[j]
                    w_nores -= ((w_nores[:, j] / d).unsqueeze(1)).matmul(row.unsqueeze(0))
                    mask[j] = True
                    w_nores[:, mask] = 0
                    row /= torch.sqrt(d)
                    Hinvs_nores -= row.unsqueeze(1).matmul(row.unsqueeze(0))
                    if dropped + 1 == num_threshold:
                        break
            else:
                res_mask = torch.zeros(self.d_col // headsize, device=self.W_device).bool()
                res_mask[self.neuron_indices['residual']] = True
                res_mask = res_mask.repeat_interleave(headsize)
                mask1 = torch.zeros(count, device=device).bool()
                for dropped in range(count + 1):
                    blocks = Hinvs_nores.reshape(count, headsize, count, headsize)
                    blocks = blocks[rangecount, :, rangecount, :]
                    try:
                        invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                    except:
                        invblocks = torch.linalg.pinv(blocks, hermitian=True)
                    W1 = w_nores.reshape((self.d_row, count, headsize)).transpose(0, 1)
                    lambd = torch.bmm(W1, invblocks)
                    scores = torch.sum(lambd * W1, (1, 2))
                    scores[mask1] = float('inf')
                    j = torch.argmin(scores)
                    rows = Hinvs_nores[(headsize * j):(headsize * (j + 1)), :]
                    d = invblocks[j]
                    w_nores -= lambd[j].matmul(rows)
                    mask1[j] = True
                    mask[(headsize * j):(headsize * (j + 1))] = True
                    w_nores[:, mask] = 0
                    Hinvs_nores -= rows.t().matmul(d.matmul(rows))
                    Hinvs_nores[rangecolumns[mask], rangecolumns[mask]] = 1
                    if dropped + 1 == num_threshold:
                        break
            origW[:,~res_mask] = w_nores
            totalmask[~res_mask] = mask
        else:
            raise ValueError(f"Invalid group method: {self.group_method}")
        return origW.to(dtype), totalmask

    def prune_struct(self, pruning_ratio, headsize: int = 1):
        self.pruning_pre_step()
        sparse_weights, mask = self.step(pruning_ratio, headsize)
        return sparse_weights, mask

    @torch.no_grad()
    def _prepare(self):
        ws = [None] * len(self.Hs)
        Hinvs = [None] * len(self.Hs)
        for k in range(len(self.Hs)):
            # get columns with all zeros
            ws[k] = self.W[:, self.neuron_indices[k]]
            zero_cols = torch.nonzero(ws[k].eq(0).all(dim=0))
            H = self.Hs[k]
            # mask rows with zero input channels
            H[zero_cols, :] = 0
            H[:, zero_cols] = 0
            H[zero_cols, zero_cols] = 1
            # invert
            H = inv_sym(H)
            ###H_inv_cho = torch.linalg.cholesky(H, upper=True)
            Hinvs[k] = H
        return ws, Hinvs
    
    @torch.no_grad()
    def _prepare_res(self):
        res_mask = torch.zeros(self.d_col, device=self.W_device).bool()
        res_mask[self.neuron_indices['residual']] = True
        w_nores = self.W[:, ~res_mask]
        H_nores = self.H[~res_mask,:][:,~res_mask]
        zero_cols = torch.nonzero(w_nores.eq(0).all(dim=0))
        # mask rows with zero input channels
        H_nores[zero_cols, :] = 0
        H_nores[:, zero_cols] = 0
        H_nores[zero_cols, zero_cols] = 1
        # invert
        Hinvs_nores = inv_sym(H_nores)
        ###H_inv_cho = torch.linalg.cholesky(H, upper=True)
        return w_nores, Hinvs_nores
    
class FastOBCStruct_noiter:

    def __init__(self, layer: nn.Module, rel_damp: float = 1e-2, is_attn: bool = False, verbose: bool = False):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = layer.weight.shape[0], np.prod(layer.weight.shape[1:])
        # FastOBC hyperparameters
        self.rel_damp = rel_damp
        # backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init hessian
        self.H = None
        self.num_samples = 0
        # misc args
        self.verbose = verbose
        self.is_attn = is_attn

    @staticmethod
    def _validate_layer(layer):
        assert isinstance(layer, (nn.Linear, _ConvNd)), "FastOBC supports only linear and convolutional layers."

    # preparatory methods
    @torch.no_grad()
    def update(self, input: Tensor) -> None:
        """
        Update the estimate of Hessian matrix from a batch of data.

        Args:
            input: batch of layer inputs
        """
        # get batch size
        batch_size = input.shape[0]
        # init hessian
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        # input reshaping
        if isinstance(self.layer, nn.Linear):
            input = input.reshape(-1, input.shape[-1])
        else:
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            # output size (batch_size, channels * \prod kernel_size, num_patches)
            input = unfold(input)
            input = input.transpose(1, 2).flatten(0, 1)
        # cast input to float32 before addition
        input = input.float()
        # hessian update
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        self.H.addmm_(input.T, input, beta=beta, alpha=alpha)
        # update number of collected samples
        self.num_samples += batch_size

    def reset(self) -> None:
        self.W = None
        self.H = None
        self.num_samples = 0
        torch.cuda.empty_cache()

    @torch.no_grad()
    def pruning_pre_step(self) -> None:
        """
        Preparatory step with hessian regularization and weight reshaping.
        """
        assert self.H is not None, "One has to process at least one sample of calibration data to run pruning"
        damp = self.rel_damp * torch.diag(self.H).mean()
        self.H[range(self.d_col), range(self.d_col)] += damp
        self.W = self.W.clone().float()
        if isinstance(self.layer, _ConvNd):
            self.W = self.W.flatten(1, -1)

    # mostly copy pasted from ZipLM prune_struct method
    # I assumethis can be significantly optimized by not iterating over every single column to remove (do in blocks)
    def step(self, pruning_ratio: float, headsize: int) -> List[Tensor]:
        d_col, device, dtype = self.d_col, self.W_device, self.W_dtype
        torch.cuda.empty_cache()
        # prepare weight and Cholesky of H^{-1}
        W, Hinv = self._prepare()

        count = d_col // headsize
        num_threshold = int(pruning_ratio * count)
        mask = torch.zeros(count, device=device).bool()
        rangecount = torch.arange(count, device=device)

        if headsize == 1:
            diag = torch.diagonal(Hinv)
            scores = torch.sum(W ** 2, 0) / diag
            _, indices = torch.topk(scores, num_threshold, largest=False)
            mask[indices] = True
            
        else:
            blocks = Hinv.reshape(count, headsize, count, headsize)
            blocks = blocks[rangecount, :, rangecount, :]
            try:
                invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
            except:
                invblocks = torch.linalg.pinv(blocks, hermitian=True)
            W1 = W.reshape((self.d_row, count, headsize)).transpose(0, 1)
            lambd = torch.bmm(W1, invblocks)
            scores = torch.sum(lambd * W1, (1, 2))
            _, indices = torch.topk(scores, num_threshold, largest=False)
            mask[indices] = True
 
        return mask

    def prune_struct(self, pruning_ratio, headsize: int = 1) -> List[Tensor]:
        self.pruning_pre_step()
        mask = self.step(pruning_ratio, headsize)
        return mask
    
    def reprune_struct(self, mask) -> List[Tensor]:

        d_col, device, dtype = self.d_col, self.W_device, self.W_dtype
        torch.cuda.empty_cache()
        # prepare weight and Cholesky of H^{-1}
        W, Hinv = self._prepare()
        indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        E = torch.eye(len(mask), device=mask.device)[:,indices]
        invE = Hinv[indices, :][:, indices]
        try:
            inv_invE = torch.cholesky_inverse(torch.linalg.cholesky(invE))
        except:
            inv_invE = torch.linalg.pinv(invE, hermitian=True)
        delta = Hinv @ E @ inv_invE @ torch.transpose(W[:, indices], 0, 1)
        sparse_weights = W - delta.t()
        return sparse_weights.to(dtype)


    @torch.no_grad()
    def _prepare(self):
        w = self.W
        # get columns with all zeros
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        H = self.H
        # mask rows with zero input channels
        H[zero_cols, :] = 0
        H[:, zero_cols] = 0
        H[zero_cols, zero_cols] = 1
        # invert
        H = inv_sym(H)
        ###H_inv_cho = torch.linalg.cholesky(H, upper=True)
        return w, H #H_inv_cho
    
class FastOBCStruct_noitergroup(FastOBCStruct_group):

    def __init__(self, layer: nn.Module, rel_damp: float = 1e-2, is_attn: bool = False, headsize: int = 1, verbose: bool = False, group_num: int = 7, neuron_indices: list = None, group_method: str = 'group_res'):
        super().__init__(layer, rel_damp, is_attn, headsize, verbose, group_num, neuron_indices, group_method)
    
    def step(self, pruning_ratio: float, headsize: int):
        device, dtype = self.W_device, self.W_dtype
        torch.cuda.empty_cache()
        # prepare weight and Cholesky of H^{-1}
        origW = self.W.clone()
        totalmask = torch.zeros(self.d_col, device=device).bool()
        if self.group_method == 'group_res' or self.group_method == 'onlyres':
            pruning_ratio = pruning_ratio * self.group_num / (self.group_num - 1)
        if self.group_method == 'group_res' or self.group_method == 'group_nores':
            Ws, Hinvs = self._prepare()
            for k in range(len(Hinvs)):
                W, Hinv = Ws[k], Hinvs[k]
                d_col = W.shape[1]

                count = d_col // headsize
                mask = torch.zeros(d_col, device=device).bool()
                rangecount = torch.arange(count, device=device)
                rangecolumns = torch.arange(d_col, device=device)
                neuron_indices = self.neuron_indices[k]

                if headsize == 1:
                    diag = torch.diagonal(Hinv)
                    group_W_metric = torch.sum(W ** 2, 0) / diag
                    thresh = torch.sort(group_W_metric.cuda())[0][int(group_W_metric.numel()*pruning_ratio)].cpu()
                    mask = (group_W_metric<thresh)

                    E = torch.eye(len(mask), device=mask.device)[:,mask]
                    invE = Hinv[mask, :][:, mask]
                    try:
                        inv_invE = torch.cholesky_inverse(torch.linalg.cholesky(invE))
                    except:
                        inv_invE = torch.linalg.pinv(invE, hermitian=True)
                    delta = Hinv @ E @ inv_invE @ torch.transpose(W[:, mask], 0, 1)
                    W = W - delta.t()
                else:
                    mask1 = torch.zeros(count, device=device).bool()
                    blocks = Hinv.reshape(count, headsize, count, headsize)
                    blocks = blocks[rangecount, :, rangecount, :]
                    try:
                        invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                    except:
                        invblocks = torch.linalg.pinv(blocks, hermitian=True)
                    W1 = W.reshape((self.d_row, count, headsize)).transpose(0, 1)
                    lambd = torch.bmm(W1, invblocks)
                    group_W_metric = torch.sum(lambd * W1, (1, 2))
                    _, indices = torch.topk(group_W_metric, int(group_W_metric.numel()*pruning_ratio), largest=False)
                    mask1[indices] = True
                    mask = mask1.repeat_interleave(headsize)

                    E = torch.eye(len(mask), device=mask.device)[:,mask]
                    invE = Hinv[mask, :][:, mask]
                    try:
                        inv_invE = torch.cholesky_inverse(torch.linalg.cholesky(invE))
                    except:
                        inv_invE = torch.linalg.pinv(invE, hermitian=True)
                    delta = Hinv @ E @ inv_invE @ torch.transpose(W[:, mask], 0, 1)
                    W = W - delta.t()

                origW[:,neuron_indices] = W
                totalmask[neuron_indices] = mask

        elif self.group_method == 'onlyres':
            w_nores, Hinvs_nores = self._prepare_res() 
            d_col = w_nores.shape[1]
            count = d_col // headsize
            mask = torch.zeros(d_col, device=device).bool()
            rangecount = torch.arange(count, device=device)
            rangecolumns = torch.arange(d_col, device=device)
            res_mask = torch.zeros(self.d_col, device=self.W_device).bool()
            res_mask[self.neuron_indices['residual']] = True

            if headsize == 1:
                res_mask = torch.zeros(self.d_col, device=self.W_device).bool()
                res_mask[self.neuron_indices['residual']] = True
                diag = torch.diagonal(Hinvs_nores)
                scores = torch.sum(w_nores ** 2, 0) / diag
                thresh = torch.sort(scores.cuda())[0][int(scores.numel()*pruning_ratio)].cpu()
                mask = (scores<thresh)

                E = torch.eye(len(mask), device=mask.device)[:,mask]
                invE = Hinvs_nores[mask, :][:, mask]
                try:
                    inv_invE = torch.cholesky_inverse(torch.linalg.cholesky(invE))
                except:
                    inv_invE = torch.linalg.pinv(invE, hermitian=True)
                delta = Hinvs_nores @ E @ inv_invE @ torch.transpose(w_nores[:, mask], 0, 1)
                w_nores = w_nores - delta.t()
            else:
                res_mask = torch.zeros(self.d_col // headsize, device=self.W_device).bool()
                res_mask[self.neuron_indices['residual']] = True
                res_mask = res_mask.repeat_interleave(headsize)
                mask1 = torch.zeros(count, device=device).bool()
                blocks = Hinvs_nores.reshape(count, headsize, count, headsize)
                blocks = blocks[rangecount, :, rangecount, :]
                try:
                    invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                except:
                    invblocks = torch.linalg.pinv(blocks, hermitian=True)
                W1 = w_nores.reshape((self.d_row, count, headsize)).transpose(0, 1)
                lambd = torch.bmm(W1, invblocks)
                scores = torch.sum(lambd * W1, (1, 2))
                _, indices = torch.topk(scores, int(scores.numel()*pruning_ratio), largest=False)
                mask1[indices] = True
                mask = mask1.repeat_interleave(headsize)

                if mask.sum() == len(mask):
                    w_nores = torch.zeros_like(w_nores)
                else:
                    E = torch.eye(len(mask), device=mask.device)[:,mask]
                    invE = Hinvs_nores[mask, :][:, mask]
                    try:
                        inv_invE = torch.cholesky_inverse(torch.linalg.cholesky(invE))
                    except:
                        inv_invE = torch.linalg.pinv(invE, hermitian=True)
                    delta = Hinvs_nores @ E @ inv_invE @ torch.transpose(w_nores[:, mask], 0, 1)
                    w_nores = w_nores - delta.t()

            origW[:,~res_mask] = w_nores
            totalmask[~res_mask] = mask
        else:
            raise ValueError(f"Invalid group method: {self.group_method}")
        return origW.to(dtype), totalmask

class FastOBCStruct_noitergroupweight(FastOBCStruct_noitergroup):

    def __init__(self, layer: nn.Module, rel_damp: float = 1e-2, is_attn: bool = False, headsize: int = 1, verbose: bool = False, group_num: int = 7, neuron_indices: list = None, group_method: str = 'group_res', cluster_id: Tensor = None, dist_matrix: Tensor = None, tau: float = 1):
        super().__init__(layer, rel_damp, is_attn, headsize, verbose, group_num, neuron_indices, group_method)
        self.cluster_id = cluster_id
        self.dist_matrix = dist_matrix
        self.tau = tau
    # preparatory methods
    @torch.no_grad()
    def update(self, input: Tensor) -> None:
        """
        Update the estimate of Hessian matrix from a batch of data.

        Args:
            input: batch of layer inputs
        """
        # get batch size
        batch_size = input.shape[0]
        if batch_size > 1:
            raise NotImplementedError("Batch size > 1 is not supported")
        # init hessian
        if self.group_method == 'group_res' or self.group_method == 'group_nores':
            for i in range(len(self.Hs)):
                if self.Hs[i] is None:
                    d_col = len(self.neuron_indices[i])
                    self.Hs[i] = torch.zeros((d_col, d_col), device=input.device, dtype=torch.float32)
        else:
            if self.H is None:
                self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        # input reshaping
        if isinstance(self.layer, nn.Linear):
            input = input.reshape(-1, input.shape[-1])
        else:
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            # output size (batch_size, channels * \prod kernel_size, num_patches)
            input = unfold(input)
            input = input.transpose(1, 2).flatten(0, 1)
        # cast input to float32 before addition
        input = input.float()
        # hessian update
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        if self.group_method == 'group_res' or self.group_method == 'group_nores':
            this_sample_id = self.cluster_id[self.num_samples]

            minlength = len(self.Hs)
            counts = torch.bincount(this_sample_id, minlength=minlength)
            sum_by_class = torch.zeros(minlength, input.shape[1], device=input.device)
            sum_by_class.index_add_(0, this_sample_id, input)
            centers = sum_by_class / counts.clamp(min=1).unsqueeze(1)
            dist_matrix = torch.cdist(centers, centers, p=2)
            valid_mask = counts > 0
            dist_matrix[~valid_mask] = 0

            for k in range(len(self.Hs)):
                neuron_indices = self.neuron_indices[k]
                group_weights = dist_matrix[k].to(input.dtype)
                
                w = group_weights[this_sample_id]
                w = torch.nn.functional.softmax(-w/self.tau, dim=0)
                scaled_w = w * input.shape[0]
                
                weighted_input = input[:,neuron_indices] * scaled_w.sqrt().unsqueeze(1)

                self.Hs[k].addmm_(weighted_input.T, weighted_input, beta=beta, alpha=alpha)
        else:
            raise ValueError(f"Weighted method cannot be used in onlyres!")
        self.num_samples += batch_size