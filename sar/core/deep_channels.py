import torch.nn as nn


class DeepSets(nn.Module):
    def __init__(self, phi, rho) -> None:
        # A version of DeepSets: https://arxiv.org/abs/1703.06114
        super().__init__()
        self.phi = phi
        self.rho = rho
    
    def forward(self, x):
        x_emb = self.phi(x)
        set_emb = x_emb.sum(dim=-2, keepdim=True)
        set_emb = self.rho(set_emb)
        return set_emb


class Phi(nn.Module):
    def __init__(self, input_dim, n_layer) -> None:
        super().__init__()
        module_list = nn.ModuleList()
        for i in range(n_layer):
            module_list.append(nn.Linear(input_dim, input_dim))
            if i < n_layer - 1:
                module_list.append(nn.ReLU())
        self.project = nn.Sequential(module_list)
    
    def forward(self, x):
        return self.project(x)


class Compressor(nn.Module):
    def __init__(self, feature_dim, n_kernel) -> None:
        super().__init__()
        self.n_kernel = n_kernel
        phi_set = [Phi(feature_dim, n_layer=3) for _ in range(self.n_kernel)]
        rho_set = [nn.Linear(feature_dim, feature_dim) for _ in range(self.n_kernel)]
        self.out_ports = [DeepSets(phi, rho) for phi, rho in zip(phi_set, rho_set)]
    
    def forward(self, x):
        output = [self.out_ports(x) for _ in range(self.n_kernel)].cat(dim=-2)
        return output

