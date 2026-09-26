import torch
import torch.nn as nn


class CategoricalEncoder(nn.Module):
    def __init__(self, cardinalities, d_embedding, bias=True):
        super().__init__()

        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, d_embedding) for n in cardinalities]
        )
        self.bias = nn.Parameter(
            torch.empty(len(cardinalities), d_embedding)
        ) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        bound = self.embeddings[0].embedding_dim ** -0.5
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.stack([emb(x[..., i]) for i, emb in enumerate(self.embeddings)], dim=-2)
        return x + self.bias if self.bias is not None else x