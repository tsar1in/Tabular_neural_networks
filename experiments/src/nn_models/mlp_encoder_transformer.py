import torch
import torch.nn as nn


class MlpEncoder(nn.Module):
    def __init__(self, categorical_features, numerical_features, lookup_layers, embedding_dim):
        super().__init__()
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.embedding_dim = embedding_dim
        self.lookup_layers = lookup_layers
        
        self.cat_embeddings = nn.ModuleDict()
        for key in categorical_features:
            self.cat_embeddings[key] = nn.Embedding(
                num_embeddings=lookup_layers[key]['vocab_size'],
                embedding_dim=embedding_dim
            )
        
        self.num_embeddings = nn.ModuleDict()
        for key in numerical_features:
            self.num_embeddings[key] = nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.ReLU()
            )

    def forward(self, x_cat, x_num):
        embeddings = []
        
        for i, key in enumerate(self.categorical_features):
            embeddings.append(
                self.cat_embeddings[key](x_cat[:, i].long()).unsqueeze(1)
            )

        for i, key in enumerate(self.numerical_features):
            num_feature = (x_num[:, i].float() - self.lookup_layers[key]['mean']) / self.lookup_layers[key]['std']
            num_feature = num_feature.unsqueeze(-1)
            embeddings.append(
                self.num_embeddings[key](num_feature).unsqueeze(1)
            )
        
        return torch.cat(embeddings, dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, tf_dropout, ff_dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=tf_dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        
        self.fcn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        
        fcn_output = self.fcn(x)
        x = x + fcn_output
        x = self.norm2(x)
        
        return x


class MlpEncoderTransformer(nn.Module):
    def __init__(self, categorical_features, numerical_features, lookup_layers,
                 tf_dropout_rates, ff_dropout_rates,
                 mlp_hidden_units_factors=[2, 1], mlp_dropout_rates=[0., 0.],
                 embedding_dim=12, num_transformer_blocks=6, num_heads=3):
        super().__init__()
        
        self.feature_encoder = MlpEncoder(
            categorical_features, numerical_features, lookup_layers, embedding_dim
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                tf_dropout=tf_dropout_rates[i],
                ff_dropout=ff_dropout_rates[i]
            )
            for i in range(num_transformer_blocks)
        ])
        
        mlp_hidden_units = [
            int(factor * embedding_dim) for factor in mlp_hidden_units_factors
        ]
        
        mlp_layers = []
        for i, units in enumerate(mlp_hidden_units):
            mlp_layers.extend([
                nn.BatchNorm1d(embedding_dim),
                nn.Linear(embedding_dim, units),
                nn.SELU(),
                nn.Dropout(mlp_dropout_rates[i])
            ])
            embedding_dim = units
        
        mlp_layers.append(nn.Linear(embedding_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x_cat, x_num):
        x = self.feature_encoder(x_cat, x_num)
    
        for block in self.transformer_blocks:
            x = block(x)
        
        x = x[:, 0]  # Берем первый токен (CLS)
        
        return self.mlp(x)