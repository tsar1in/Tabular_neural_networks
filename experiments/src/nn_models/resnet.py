from torch import nn
from .nn_modules import get_normalization_modules, get_activation_functions


class ResNetBlock(nn.Module):
    def __init__(self, main_dim, hidden_dim, normalization='BatchNorm1d', activation='ReLU',
                 dropout_rate_first=0.0, dropout_rate_second=0.0, skip_connection=True):
        super(ResNetBlock, self).__init__()

        self.norm = get_normalization_modules(normalization, main_dim)
        self.fc1 = nn.Linear(main_dim, hidden_dim)
        self.activation = get_activation_functions(activation)
        self.dropout1 = nn.Dropout(dropout_rate_first)
        self.fc2 = nn.Linear(hidden_dim, main_dim)
        self.dropout2 = nn.Dropout(dropout_rate_second)
        self.skip_connection = skip_connection

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        if self.skip_connection:
            x = x + residual

        return x


class ResNetHead(nn.Module):
    def __init__(self, main_dim, output_dim, normalization='BatchNorm1d', activation='ReLU'):
        super(ResNetHead, self).__init__()

        self.norm = get_normalization_modules(normalization, main_dim)
        self.activation = get_activation_functions(activation)
        self.linear = nn.Linear(main_dim, output_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.linear(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_dim, num_blocks, main_dim, hidden_dim, output_dim, normalization='BatchNorm1d',
                activation='ReLU', dropout_first=0.0, dropout_second=0.0, skip_connection=True):
        super(ResNet, self).__init__()

        self.input_layer = nn.Linear(input_dim, main_dim)
        self.layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.layers.append(
                ResNetBlock(
                    main_dim,
                    hidden_dim,
                    normalization=normalization,
                    activation=activation,
                    dropout_rate_first=dropout_first,
                    dropout_rate_second=dropout_second,
                    skip_connection=skip_connection
                )
            )

        self.head = ResNetHead(
            main_dim,
            output_dim,
            normalization=normalization,
            activation=activation
        )

    def forward(self, x):
        x = self.input_layer(x)

        for layer in self.layers:
            x = layer(x)

        x = self.head(x)

        return x