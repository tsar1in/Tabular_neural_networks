from torch import nn
import torch.nn.functional as F


class ReGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)

def get_activation_functions(activation):
    activation_functions = {
        'ReLu': nn.ReLU(),
        'SeLu': nn.SELU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'lReLu': nn.LeakyReLU(),
        'ReGLU': ReGLU(),
    }

    return activation_functions[activation]

def get_normalization_modules(normalization, dim):
    normalization_modules = {
        "LayerNorm": nn.LayerNorm(dim),
        "BatchNorm1d": nn.BatchNorm1d(dim),
        "BatchNorm2d": nn.BatchNorm2d(dim),
        "BatchNorm3d": nn.BatchNorm3d(dim),
        "InstanceNorm1d": nn.InstanceNorm1d(dim),
        "InstanceNorm2d": nn.InstanceNorm2d(dim),
        "InstanceNorm3d": nn.InstanceNorm3d(dim)
    }

    return normalization_modules[normalization]

