import torch
import torch.nn as nn

class clusterGPT:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, cluster_center, token_mean = None, pca_proj = None):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.out_dim = layer.weight.data.shape[0]
        self.in_dim = layer.weight.data.shape[1]
        self.nsamples = 0
        self.cluster_center = cluster_center
        self.token_mean = token_mean
        self.pca_proj = pca_proj

        self.cluster_id = []

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))

        if self.pca_proj == None:

            distances = (inp.unsqueeze(1) - self.cluster_center.unsqueeze(0)).pow(2).sum(dim=2)
            cluster_indices = torch.argmin(distances, dim=1)
            self.cluster_id.append(cluster_indices.unsqueeze(0))

        else:

            inp_pca = (inp - self.token_mean.to(inp.dtype)) @ self.pca_proj.to(inp.dtype)
            distances = (inp_pca.unsqueeze(1) - self.cluster_center.unsqueeze(0)).pow(2).sum(dim=2)
            cluster_indices = torch.argmin(distances, dim=1)
            self.cluster_id.append(cluster_indices.unsqueeze(0))

        
    def free(self):
        self.cluster_center = None
        self.cluster_id = None
        self.token_mean = None
        self.pca_proj = None
        torch.cuda.empty_cache()  