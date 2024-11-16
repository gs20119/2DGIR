
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        dim = self.kwargs['input_dims']
        out_dim = 0
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += dim

        if self.kwargs['log_sampling']: 
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else: freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += dim

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dim=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dim,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


# Activations 
class ExpActivation(nn.Module):
    def __init__(self, max_light=5.0):
        super().__init__()
        self.max_light=max_light

    def forward(self, x):
        return torch.exp(torch.clamp(x, max=self.max_light))


# Direct Light Modules
class ViewOnlyNeRF(nn.Module): # from NeRO implementation
    def __init__(self, input_dim=2, output_dim=3, run_dim=256, exp_max=5.0) -> object:
        super(ViewOnlyNeRF, self).__init__()
        self.dir_enc, dir_dim = get_embedder(6, input_dim) # need config for multires
        self.module = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(dir_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
        )
        self.activation = ExpActivation(max_light=exp_max)
    
    def forward(self, input_views): # view = from point to sphere
        v = self.dir_enc(input_views)
        x = self.module(v)
        return self.activation(x)