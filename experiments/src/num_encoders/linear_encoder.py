import torch
import torch.nn as nn


class LinearEncoder(nn.Module):
    def __init__(self, n_features, d_embedding):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.bias = nn.Parameter(torch.empty(n_features, d_embedding))
        self.reset_parameters()

    def reset_parameters(self):
        bound = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias