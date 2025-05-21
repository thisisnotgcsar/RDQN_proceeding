from typing import Sequence
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class QFunc(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: Sequence[int],
                 output_size: int,
                 activation: str = 'tanh'):
        super().__init__()
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError(f'Activation function {activation} not supported')
        layers = [nn.Linear(input_size, hidden_size[0])]
        for dim in hidden_size[1:]:
            layers.extend([
                activation_fn,
                nn.Linear(layers[-1].out_features, dim),
            ])
        layers.append(activation_fn)
        layers.append(nn.Linear(hidden_size[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
