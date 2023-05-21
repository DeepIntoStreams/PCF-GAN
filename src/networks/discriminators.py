import torch.nn as nn
import torch
from torch import nn
from typing import Tuple
from src.utils import init_weights


class LSTMDiscriminator(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, n_layers: int, out_dim, return_seq=False
    ):
        super(LSTMDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.model = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_dim, out_dim)

        self.model.apply(init_weights)
        self.linear1.apply(init_weights)
        self.linear.apply(init_weights)
        self.return_seq = return_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.LeakyReLU()(self.linear1(x))

        if self.return_seq:
            h = self.model(x)[0]
        else:
            h = self.model(x)[0][:, -1:]

        x = self.linear(nn.Tanh()(h))
        return x
