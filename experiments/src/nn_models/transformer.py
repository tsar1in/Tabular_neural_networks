import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.parameter import Parameter
from ..nn_models.nn_modules import ReGLU
from ..num_encoders.linear_encoder import LinearEncoder
from ..cat_encoder.default_cat_encoder import CategoricalEncoder


class MultiheadAttention(nn.Module):
    def __init__(self, *, d_embedding, n_heads, dropout, n_tokens=None, 
                linformer_kv_compression_ratio=None, linformer_kv_compression_sharing=None
    ):
        super().__init__()

        self._n_heads = n_heads
        self.W_q = nn.Linear(d_embedding, d_embedding)
        self.W_k = nn.Linear(d_embedding, d_embedding)
        self.W_v = nn.Linear(d_embedding, d_embedding)
        self.W_out = nn.Linear(d_embedding, d_embedding) if n_heads > 1 else None
        self.dropout = nn.Dropout(dropout) if dropout else None

        if linformer_kv_compression_ratio is not None:
            n_compressed = max(1, int(n_tokens * linformer_kv_compression_ratio))
            self.key_compression = nn.Linear(
                n_tokens, 
                n_compressed, 
                bias=False
            )
            self.value_compression = nn.Linear(
                n_tokens, 
                n_compressed, 
                bias=False
            ) if linformer_kv_compression_sharing == 'headwise' else None
        else:
            self.key_compression = self.value_compression = None

        for m in [self.W_q, self.W_k, self.W_v]:
            nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self._n_heads
        return x.view(batch_size, n_tokens, self._n_heads, d_head).transpose(1, 2).reshape(batch_size * self._n_heads, n_tokens, d_head)

    def forward(self, x_q, x_kv):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)

        if self.key_compression is not None:
            k = self.key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = (self.key_compression if self.value_compression is None else self.value_compression)(v.transpose(1, 2)).transpose(1, 2)

        q, k, v = self._reshape(q), self._reshape(k), self._reshape(v)
        attention_logits = (q @ k.transpose(1, 2)) / sqrt(k.size(-1))
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)

        x = attention_probs @ v
        x = x.view(-1, self._n_heads, x.size(1), x.size(-1)).transpose(1, 2).flatten(2)
        return self.W_out(x) if self.W_out is not None else x
    

class FTTransformerBackbone(nn.Module):
    def __init__(self, *, d_out, n_blocks, d_block, attention_n_heads, attention_dropout,
        ffn_d_hidden=None, ffn_d_hidden_multiplier=None, ffn_dropout, ffn_activation='ReGLU',
        residual_dropout, n_tokens=None, linformer_kv_compression_ratio=None, linformer_kv_compression_sharing=None
    ):
        super().__init__()

        if ffn_d_hidden is None and ffn_d_hidden_multiplier is not None:
            ffn_d_hidden = int(d_block * ffn_d_hidden_multiplier)

        _ffn_d_hidden = ffn_d_hidden
        ffn_use_reglu = (ffn_activation == 'ReGLU')
        
        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            modules = OrderedDict()

            if i > 0:
                modules['attention_normalization'] = nn.LayerNorm(d_block)

            modules['attention'] = MultiheadAttention(
                d_embedding=d_block,
                n_heads=attention_n_heads,
                dropout=attention_dropout,
                n_tokens=n_tokens,
                linformer_kv_compression_ratio=linformer_kv_compression_ratio,
                linformer_kv_compression_sharing=linformer_kv_compression_sharing,
            )
            modules['attention_residual_dropout'] = nn.Dropout(
                residual_dropout
            )
            modules['ffn_normalization'] = nn.LayerNorm(
                d_block
            )
            modules['ffn'] = self.layers_to_sequential(
                ('linear1', nn.Linear(
                    d_block, 
                    _ffn_d_hidden * (2 if ffn_use_reglu else 1)
                )),
                ('activation', ReGLU() if ffn_use_reglu else nn.ReLU()),
                ('dropout', nn.Dropout(
                    ffn_dropout
                )),
                ('linear2', nn.Linear(
                    _ffn_d_hidden, 
                    d_block
                )),
            )
            modules['ffn_residual_dropout'] = nn.Dropout(
                residual_dropout
            )
            modules['output'] = nn.Identity()
            self.blocks.append(nn.ModuleDict(modules))

        self.output = (
            None if d_out is None else
            self.layers_to_sequential(
                ('normalization', nn.LayerNorm(d_block)),
                ('activation', nn.ReLU()),
                ('linear', nn.Linear(d_block, d_out)),
            )
        )

    def forward(self, x):
        n_blocks = len(self.blocks)
        for i_block, block_module_dict in enumerate(self.blocks):
            x_identity = x
            
            _x_normalized_or_original = x
            if 'attention_normalization' in block_module_dict:
                 _x_normalized_or_original = block_module_dict['attention_normalization'](x)
            
            _q_data = _x_normalized_or_original[:, :1] if i_block + 1 == n_blocks else _x_normalized_or_original
            _kv_data = _x_normalized_or_original
            
            _attention_output = block_module_dict['attention'](_q_data, _kv_data)
            _attention_output = block_module_dict['attention_residual_dropout'](_attention_output)
            x = x_identity + _attention_output

            x_identity = x
            _x_normalized = block_module_dict['ffn_normalization'](x)
            _ffn_output = block_module_dict['ffn'](_x_normalized)
            _ffn_output = block_module_dict['ffn_residual_dropout'](_ffn_output)
            x = x_identity + _ffn_output
            
            x = block_module_dict['output'](x)

        x = x[:, 0] # берем <CLS> токен
        if self.output is not None:
            x = self.output(x)
        return x
    
    @staticmethod
    def layers_to_sequential(*modules):
        return nn.Sequential(OrderedDict(modules))


class _CLSEmbedding(nn.Module):
    def __init__(self, d_embedding):
        super().__init__()
        self.weight = Parameter(
            torch.empty(d_embedding)
        )
        self.reset_parameters()

    def reset_parameters(self):
        bound = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, batch_dims):
        return self.weight.expand(*batch_dims, 1, -1)


class FTTransformer(nn.Module):
    def __init__(self, *, n_cont_features, cat_cardinalities, **backbone_kwargs):
        super().__init__()

        d_block: int = backbone_kwargs['d_block']
        self.cls_embedding = _CLSEmbedding(d_block)

        self.cont_embeddings = (
            LinearEncoder(n_cont_features, d_block) if n_cont_features > 0 else None
        )
        self.cat_embeddings = (
            CategoricalEncoder(cat_cardinalities, d_block, True)
            if cat_cardinalities
            else None
        )
        
        _n_tokens_for_linformer = 1 + n_cont_features + len(cat_cardinalities)
        _backbone_n_tokens = (
            _n_tokens_for1_linformer
            if backbone_kwargs.get('linformer_kv_compression_ratio') is not None
            else None
        )

        self.backbone = FTTransformerBackbone(
            **backbone_kwargs,
            n_tokens=_backbone_n_tokens,
        )

    def forward(self, x_cont, x_cat):
        _current_batch_dims = (
            x_cont if x_cont is not None else x_cat
        ).shape[:-1]

        x_embeddings = []
        x_embeddings.append(self.cls_embedding(_current_batch_dims))

        if self.cont_embeddings is not None and x_cont is not None:
            x_embeddings.append(self.cont_embeddings(x_cont))
        
        if self.cat_embeddings is not None and x_cat is not None:
            x_embeddings.append(self.cat_embeddings(x_cat))
        
        x = torch.cat(x_embeddings, dim=1)
        x = self.backbone(x)
        return x