from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def mlp(inp_dim, out_dim, hidden_sizes, bias=True):
    if len(hidden_sizes) == 0:
        return nn.Linear(inp_dim, out_dim, bias=bias)
    model = [nn.Linear(inp_dim, hidden_sizes[0], bias=bias)]
    model += [nn.ReLU(inplace=True)]
    for i in range(1, len(hidden_sizes)):
        model += [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], bias=bias)]
        model += [nn.ReLU(inplace=True)]
    if out_dim != 0:
        model += [nn.Linear(hidden_sizes[-1], out_dim, bias=bias)]
    return nn.Sequential(*model)


class LinearNet(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int] = (),
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super(LinearNet, self).__init__()
        self.basedmodel = mlp(in_features, 1, hidden_sizes)
        if prior_scale > 0:
            self.priormodel = mlp(in_features, 1, hidden_sizes)
            for param in self.priormodel.parameters():
                param.requires_grad = False
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.basedmodel.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.xavier_normal_(param, gain=1.0)
        if self.prior_scale > 0:
            for name, param in self.priormodel.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.xavier_normal_(param, gain=1.0)

    def forward(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        out = self.basedmodel(x)
        if self.prior_scale > 0:
            prior_out = self.priormodel(x)
            out = self.posterior_scale * out + self.prior_scale * prior_out
        return out.squeeze(-1)
