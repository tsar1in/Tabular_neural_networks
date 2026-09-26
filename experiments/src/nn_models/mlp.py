from torch import nn
from .nn_modules import get_activation_functions


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim,
                 activation='ReLu', dropout_rate=0.1):
        super(MLP, self).__init__()

        layers = []
        in_features = input_dim

        for hidden_dim in hidden_layers:
            layers.append(
                nn.Linear(in_features, hidden_dim)
            )
            layers.append(
                get_activation_functions(activation)
            )
            if dropout_rate > 0:
                layers.append(
                    nn.Dropout(dropout_rate)
                )
            in_features = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(in_features, output_dim)

    def forward(self, x):
        x = self.layers(x)
        x = self.head(x)
        return x