import torch 
import torch.nn as nn 
from .layerwrapper import clusterGPT
from .data import get_loaders 
from tqdm import tqdm
from lib.fast_obc_struct import FastOBCStruct, FastOBCStruct_noiter, FastOBCStruct_noitergroup, FastOBCStruct_noitergroupweight

# create a dictionary to map the method name to the function
"""
    'IFV': Input Feature Variance
    'WIFV': Weighted Input Feature Variance
    'WIFN': Weighted Input Feature Norm
"""
metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
}


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    """
    Check the sparsity of the weights in different layers of the model.
    
    Args:
        model (nn.Module): The model to check.
        
    Returns:
        float: Ratio of the count of non-zero weights to total parameters in the model.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            sub_count += W.numel()
            count += W.numel()
            if 'self_attn' in name:
                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size
            else:
                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size
            if subset[name].bias is not None:
                count += subset[name].bias.data.numel()
                sub_count += subset[name].bias.data.numel()
            
        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


def prepare_calibration_input(model, dataloader, device, args):
    """
    Prepare inputs for model calibration. 
    
    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded. 
        
    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
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

def ziplm_prune(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print("loading calibdation data")
    dataloader, _ = get_loaders(args.prune_calibration, nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args)
    layers = model.model.layers
    headsize = model.model.config.hidden_size // model.model.config.num_attention_heads

    if args.sparsity_allocation_file != None:
        sparsity_allocation = torch.load(args.sparsity_allocation_file)

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = find_layers(layer)
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            if name == 'self_attn.o_proj':
                wrapped_layers[name] = FastOBCStruct(subset[name], rel_damp = 1e-2, is_attn = True)
            else:
                wrapped_layers[name] = FastOBCStruct(subset[name], rel_damp = 1e-2, is_attn = False)            

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].update(inp[0].data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            if args.sparsity_allocation_file != None:
                pruning_ratio_attn = sparsity_allocation['attn'][i]
                pruning_ratio_mlp = sparsity_allocation['mlp'][i]
            else:
                pruning_ratio_attn = args.pruning_ratio
                pruning_ratio_mlp = args.pruning_ratio
            if name == 'self_attn.o_proj':
                sparse_weights, mask = wrapped_layers[name].prune_struct(pruning_ratio_attn, headsize = headsize)
            else:
                sparse_weights, mask = wrapped_layers[name].prune_struct(pruning_ratio_mlp, headsize = 1)

            mask = ~mask
            if name == 'self_attn.o_proj':
                if layer.self_attn.num_key_value_groups > 1:
                    raise NotImplementedError("Not implemented for group query attention")
                retain_heads = torch.count_nonzero(mask)
                mask = mask.repeat_interleave(headsize)

                layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(mask)[0]]
                layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(mask)[0]]
                layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(mask)[0]]

                layer.self_attn.q_proj.out_features = mask.sum().item()
                layer.self_attn.k_proj.out_features = mask.sum().item()
                layer.self_attn.v_proj.out_features = mask.sum().item()
                layer.self_attn.o_proj.in_features = mask.sum().item()
            
                layer.self_attn.num_heads = retain_heads
                layer.self_attn.num_key_value_heads = retain_heads
                layer.self_attn.hidden_size = retain_heads * headsize
                layer.self_attn.o_proj.weight.data = sparse_weights[:, mask]

            else:
                layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mask)[0]]
                layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mask)[0]]
                layer.mlp.up_proj.out_features = mask.sum().item()
                layer.mlp.gate_proj.out_features = mask.sum().item()
                layer.mlp.down_proj.in_features = mask.sum().item()  
                layer.mlp.intermediate_size = mask.sum().item()
                layer.mlp.down_proj.weight.data = sparse_weights[:, mask]
            
            wrapped_layers[name].reset()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
        torch.cuda.empty_cache()

def ziplm_noiter(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print("loading calibdation data")
    dataloader, _ = get_loaders(args.prune_calibration, nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args)
    layers = model.model.layers
    headsize = model.model.config.hidden_size // model.model.config.num_attention_heads
    if args.sparsity_allocation_file != None:
        sparsity_allocation = torch.load(args.sparsity_allocation_file)

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = find_layers(layer)
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            if name == 'self_attn.o_proj':
                wrapped_layers[name] = FastOBCStruct_noiter(subset[name], rel_damp = 1e-2, is_attn = True)
            else:
                wrapped_layers[name] = FastOBCStruct_noiter(subset[name], rel_damp = 1e-2, is_attn = False)            

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].update(inp[0].data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            if args.sparsity_allocation_file != None:
                pruning_ratio_attn = sparsity_allocation['attn'][i]
                pruning_ratio_mlp = sparsity_allocation['mlp'][i]
            else:
                pruning_ratio_attn = args.pruning_ratio
                pruning_ratio_mlp = args.pruning_ratio
            if name == 'self_attn.o_proj':
                mask = wrapped_layers[name].prune_struct(pruning_ratio_attn, headsize = headsize)
                retain_heads = torch.count_nonzero(~mask)
                mask = mask.repeat_interleave(headsize)
                sparse_weights = wrapped_layers[name].reprune_struct(mask)
            else:
                mask = wrapped_layers[name].prune_struct(pruning_ratio_mlp, headsize = 1)
                sparse_weights = wrapped_layers[name].reprune_struct(mask)

            mask = ~mask
            if name == 'self_attn.o_proj':
                if layer.self_attn.num_key_value_groups > 1:
                    sparse_weights[:, ~mask] = 0
                    layer.self_attn.o_proj.weight.data = sparse_weights
                else:

                    layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(mask)[0]]
                    layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(mask)[0]]
                    layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(mask)[0]]

                    layer.self_attn.q_proj.out_features = mask.sum().item()
                    layer.self_attn.k_proj.out_features = mask.sum().item()
                    layer.self_attn.v_proj.out_features = mask.sum().item()
                    layer.self_attn.o_proj.in_features = mask.sum().item()
                
                    layer.self_attn.num_heads = retain_heads
                    layer.self_attn.num_key_value_heads = retain_heads
                    layer.self_attn.hidden_size = retain_heads * headsize
                    layer.self_attn.o_proj.weight.data = sparse_weights[:, mask]

            else:
                layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mask)[0]]
                layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mask)[0]]
                layer.mlp.up_proj.out_features = mask.sum().item()
                layer.mlp.gate_proj.out_features = mask.sum().item()
                layer.mlp.down_proj.in_features = mask.sum().item()  
                layer.mlp.intermediate_size = mask.sum().item()
                layer.mlp.down_proj.weight.data = sparse_weights[:, mask]
            
            wrapped_layers[name].reset()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
        torch.cuda.empty_cache()

def ziplm_noiter_group_prune(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print("loading calibdation data")
    dataloader, _ = get_loaders(args.prune_calibration, nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args)
    layers = model.model.layers
    headsize = model.model.config.hidden_size // model.model.config.num_attention_heads
    if args.sparsity_allocation_file != None:
        sparsity_allocation = torch.load(args.sparsity_allocation_file)
    neuron_indices = torch.load(args.neuron_indices_file)

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = find_layers(layer)
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            if name == 'self_attn.o_proj':
                if args.attn_group:
                    group_num = len(neuron_indices['attn'][i])
                    wrapped_layers[name] = FastOBCStruct_noitergroup(subset[name], rel_damp = 1e-2, is_attn = True, headsize = headsize, group_num = group_num, neuron_indices = neuron_indices['attn'][i], group_method = args.group_method)
                else:
                    wrapped_layers[name] = FastOBCStruct_noiter(subset[name], rel_damp = 1e-2, is_attn = True)
            else:
                group_num = len(neuron_indices['mlp'][i])
                wrapped_layers[name] = FastOBCStruct_noitergroup(subset[name], rel_damp = 1e-2, is_attn = False, headsize = headsize, group_num = group_num, neuron_indices = neuron_indices['mlp'][i], group_method = args.group_method)            

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].update(inp[0].data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            if args.sparsity_allocation_file != None:
                pruning_ratio_attn = sparsity_allocation['attn'][i]
                pruning_ratio_mlp = sparsity_allocation['mlp'][i]
            else:
                pruning_ratio_attn = args.pruning_ratio
                pruning_ratio_mlp = args.pruning_ratio
            if name == 'self_attn.o_proj':
                if args.attn_group:
                    sparse_weights, mask = wrapped_layers[name].prune_struct(pruning_ratio_attn, headsize = headsize)
                    retain_heads = torch.sum(~mask) // headsize
                else:
                    mask = wrapped_layers[name].prune_struct(pruning_ratio_attn, headsize = headsize)
                    retain_heads = torch.count_nonzero(~mask)
                    mask = mask.repeat_interleave(headsize)
                    sparse_weights = wrapped_layers[name].reprune_struct(mask)
            else:
                sparse_weights, mask = wrapped_layers[name].prune_struct(pruning_ratio_mlp, headsize = 1)

            mask = ~mask
            if name == 'self_attn.o_proj':
                if layer.self_attn.num_key_value_groups > 1:
                    sparse_weights[:, ~mask] = 0
                    layer.self_attn.o_proj.weight.data = sparse_weights
                else:

                    layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(mask)[0]]
                    layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(mask)[0]]
                    layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(mask)[0]]

                    layer.self_attn.q_proj.out_features = mask.sum().item()
                    layer.self_attn.k_proj.out_features = mask.sum().item()
                    layer.self_attn.v_proj.out_features = mask.sum().item()
                    layer.self_attn.o_proj.in_features = mask.sum().item()
                
                    layer.self_attn.num_heads = retain_heads
                    layer.self_attn.num_key_value_heads = retain_heads
                    layer.self_attn.hidden_size = retain_heads * headsize
                    layer.self_attn.o_proj.weight.data = sparse_weights[:, mask]

            else:
                layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mask)[0]]
                layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mask)[0]]
                layer.mlp.up_proj.out_features = mask.sum().item()
                layer.mlp.gate_proj.out_features = mask.sum().item()
                layer.mlp.down_proj.in_features = mask.sum().item()  
                layer.mlp.intermediate_size = mask.sum().item()
                layer.mlp.down_proj.weight.data = sparse_weights[:, mask]
            
            wrapped_layers[name].reset()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
        torch.cuda.empty_cache()

def ziplm_noiter_group_weight_prune(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print("loading calibdation data")
    dataloader, _ = get_loaders(args.prune_calibration, nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args)
    layers = model.model.layers
    headsize = model.model.config.hidden_size // model.model.config.num_attention_heads
    if args.sparsity_allocation_file != None:
        sparsity_allocation = torch.load(args.sparsity_allocation_file)
    neuron_indices = torch.load(args.neuron_indices_file)
    cluster_centers = torch.load(args.cluster_center_file)

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = find_layers(layer)
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})
        subset.update({'mlp.up_proj': find_layers(layer)['mlp.up_proj']})
        subset.update({'self_attn.q_proj': find_layers(layer)['self_attn.q_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)
        else:
            dev = device

        up_proj_wrapper = clusterGPT(subset['mlp.up_proj'], cluster_centers['mlp'][i].to(dev), cluster_centers['mlp_mean'][i].to(dev), cluster_centers['mlp_proj'][i].to(dev))            
        def add_batch():
            def tmp(_, inp, out):
                up_proj_wrapper.add_batch(inp[0].data, out.data)
            return tmp
        handle = subset['mlp.up_proj'].register_forward_hook(add_batch())
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        handle.remove()
        cluster_id = torch.cat(up_proj_wrapper.cluster_id)
        dist_matrix = torch.cdist(cluster_centers['mlp'][i], cluster_centers['mlp'][i], p=2)
        up_proj_wrapper.free()
        del subset['mlp.up_proj']

        if args.attn_group:
            q_proj_wrapper = clusterGPT(subset['self_attn.q_proj'], cluster_centers['attn'][i].to(dev), cluster_centers['attn_mean'][i].to(dev), cluster_centers['attn_proj'][i].to(dev))            
            def add_batch():
                def tmp(_, inp, out):
                    q_proj_wrapper.add_batch(inp[0].data, out.data)
                return tmp
            handle = subset['self_attn.q_proj'].register_forward_hook(add_batch())
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            handle.remove()
            cluster_id_attn = torch.cat(q_proj_wrapper.cluster_id)
            dist_matrix_attn = torch.cdist(cluster_centers['attn'][i], cluster_centers['attn'][i], p=2)
            up_proj_wrapper.free()
        del subset['self_attn.q_proj']

        wrapped_layers = {}
        for name in subset:
            if name == 'self_attn.o_proj':
                if args.attn_group:
                    group_num = len(neuron_indices['attn'][i])
                    wrapped_layers[name] = FastOBCStruct_noitergroupweight(subset[name], rel_damp = 1e-2, is_attn = True, headsize = headsize, group_num = group_num, neuron_indices = neuron_indices['attn'][i], cluster_id = cluster_id_attn.to(dev), dist_matrix = dist_matrix_attn.to(dev), tau = args.tau)
                else:
                    wrapped_layers[name] = FastOBCStruct_noiter(subset[name], rel_damp = 1e-2, is_attn = True)
            elif name == 'mlp.down_proj':
                group_num = len(neuron_indices['mlp'][i])
                wrapped_layers[name] = FastOBCStruct_noitergroupweight(subset[name], rel_damp = 1e-2, is_attn = False, headsize = headsize, group_num = group_num, neuron_indices = neuron_indices['mlp'][i], cluster_id = cluster_id.to(dev), dist_matrix = dist_matrix.to(dev), tau = args.tau)   
            else:
                raise ValueError("Invalid layer name") 
   
        del cluster_id
        torch.cuda.empty_cache()       

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].update(inp[0].data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            if args.sparsity_allocation_file != None:
                pruning_ratio_attn = sparsity_allocation['attn'][i]
                pruning_ratio_mlp = sparsity_allocation['mlp'][i]
            else:
                pruning_ratio_attn = args.pruning_ratio
                pruning_ratio_mlp = args.pruning_ratio
            if name == 'self_attn.o_proj':
                if args.attn_group:
                    sparse_weights, mask = wrapped_layers[name].prune_struct(pruning_ratio_attn, headsize = headsize)
                    retain_heads = torch.sum(~mask) // headsize
                else:
                    mask = wrapped_layers[name].prune_struct(pruning_ratio_attn, headsize = headsize)
                    retain_heads = torch.count_nonzero(~mask)
                    mask = mask.repeat_interleave(headsize)
                    sparse_weights = wrapped_layers[name].reprune_struct(mask)
            else:
                sparse_weights, mask = wrapped_layers[name].prune_struct(pruning_ratio_mlp, headsize = 1)

            mask = ~mask
            if name == 'self_attn.o_proj':
                if layer.self_attn.num_key_value_groups > 1:
                    sparse_weights[:, ~mask] = 0
                    layer.self_attn.o_proj.weight.data = sparse_weights
                else:

                    layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(mask)[0]]
                    layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(mask)[0]]
                    layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(mask)[0]]

                    layer.self_attn.q_proj.out_features = mask.sum().item()
                    layer.self_attn.k_proj.out_features = mask.sum().item()
                    layer.self_attn.v_proj.out_features = mask.sum().item()
                    layer.self_attn.o_proj.in_features = mask.sum().item()
                
                    layer.self_attn.num_heads = retain_heads
                    layer.self_attn.num_key_value_heads = retain_heads
                    layer.self_attn.hidden_size = retain_heads * headsize
                    layer.self_attn.o_proj.weight.data = sparse_weights[:, mask]

            else:
                layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mask)[0]]
                layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mask)[0]]
                layer.mlp.up_proj.out_features = mask.sum().item()
                layer.mlp.gate_proj.out_features = mask.sum().item()
                layer.mlp.down_proj.in_features = mask.sum().item()  
                layer.mlp.intermediate_size = mask.sum().item()
                layer.mlp.down_proj.weight.data = sparse_weights[:, mask]
            
            wrapped_layers[name].reset()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
        torch.cuda.empty_cache()