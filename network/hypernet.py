from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(input_dim, hidden_sizes, linear_layer=nn.Linear):
    model = []
    if len(hidden_sizes) > 0:
        hidden_sizes = [input_dim] + list(hidden_sizes)
        for i in range(1, len(hidden_sizes)):
            model += [linear_layer(hidden_sizes[i - 1], hidden_sizes[i])]
            model += [nn.ReLU(inplace=True)]
    model = nn.Sequential(*model)
    return model


class HyperLayer(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int,
        action_dim: int = 1,
        prior_std: float = 1.0,
        use_bias: bool = True,
        trainable: bool = True,
        out_type: str = "weight",
        weight_init: str = "xavier_normal",
        bias_init: str = "sphere-sphere",
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        assert out_type in ["weight", "bias"], f"No out type {out_type} in HyperLayer"
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.prior_std = prior_std
        self.use_bias = use_bias
        self.trainable = trainable
        self.out_type = out_type
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.device = device

        self.in_features = noise_dim
        if out_type == "weight":
            self.out_features = action_dim * hidden_dim
        elif out_type == "bias":
            self.out_features = action_dim

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        if not self.trainable:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def reset_parameters(self) -> None:
        # init weight
        if self.weight_init == "sDB":
            weight = np.random.randn(self.out_features, self.in_features).astype(
                np.float32
            )
            weight = weight / np.linalg.norm(weight, axis=1, keepdims=True)
            self.weight = nn.Parameter(
                torch.from_numpy(self.prior_std * weight).float()
            )
        elif self.weight_init == "gDB":
            weight = np.random.randn(self.out_features, self.in_features).astype(
                np.float32
            )
            self.weight = nn.Parameter(
                torch.from_numpy(self.prior_std * weight).float()
            )
        elif self.weight_init == "trunc_normal":
            bound = 1.0 / np.sqrt(self.in_features)
            nn.init.trunc_normal_(self.weight, std=bound, a=-2 * bound, b=2 * bound)
        elif self.weight_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight, gain=1.0)
        elif self.weight_init == "xavier_normal":
            nn.init.xavier_normal_(self.weight, gain=1.0)
        else:
            weight = [
                nn.init.xavier_normal_(
                    torch.zeros((self.action_dim, self.hidden_dim))
                ).flatten()
                for _ in range(self.in_features)
            ]
            self.weight = nn.Parameter(torch.stack(weight, dim=1))
            # nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # init bias
        if self.use_bias:
            if self.bias_init == "default":
                bound = 1.0 / np.sqrt(self.in_features)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                weight_bias_init, bias_bias_init = self.bias_init.split("-")
                if self.out_type == "weight":
                    if weight_bias_init == "zeros":
                        nn.init.zeros_(self.bias)
                    elif weight_bias_init == "sphere":
                        bias = np.random.randn(self.out_features).astype(np.float32)
                        bias = bias / np.linalg.norm(bias)
                        self.bias = nn.Parameter(
                            torch.from_numpy(self.prior_std * bias).float()
                        )
                    elif weight_bias_init == "xavier":
                        bias = nn.init.xavier_normal_(
                            torch.zeros((self.action_dim, self.hidden_dim))
                        )
                        self.bias = nn.Parameter(bias.flatten())
                elif self.out_type == "bias":
                    if bias_bias_init == "zeros":
                        nn.init.zeros_(self.bias)
                    elif bias_bias_init == "sphere":
                        bias = np.random.randn(self.out_features).astype(np.float32)
                        bias = bias / np.linalg.norm(bias)
                        self.bias = nn.Parameter(
                            torch.from_numpy(self.prior_std * bias).float()
                        )
                    elif bias_bias_init == "uniform":
                        bound = 1 / np.sqrt(self.hidden_dim)
                        nn.init.uniform_(self.bias, -bound, bound)
                    elif bias_bias_init == "pos":
                        bias = 1 * np.ones(self.out_features)
                        self.bias = nn.Parameter(torch.from_numpy(bias).float())
                    elif bias_bias_init == "neg":
                        bias = -1 * np.ones(self.out_features)
                        self.bias = nn.Parameter(torch.from_numpy(bias).float())

    def forward(self, z: torch.Tensor):
        z = z.to(self.device)
        return F.linear(z, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class HyperLinear(nn.Module):
    def __init__(
        self,
        noise_dim,
        out_features,
        prior_std: float = 1.0,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        use_bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        hyperlayer_params = dict(
            noise_dim=noise_dim,
            hidden_dim=out_features,
            prior_std=prior_std,
            out_type="weight",
            use_bias=use_bias,
            device=device,
        )
        self.hyper_weight = HyperLayer(
            **hyperlayer_params, trainable=True, weight_init="xavier_normal"
        )
        self.prior_weight = HyperLayer(
            **hyperlayer_params, trainable=False, weight_init="sDB"
        )

        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

    def forward(self, z, x, prior_x):
        theta = self.hyper_weight(z)
        prior_theta = self.prior_weight(z)

        if len(x.shape) > 2:
            # compute feel-good term
            out = torch.einsum("bd,bad -> ba", theta, x)
            prior_out = torch.einsum("bd,bad -> ba", prior_theta, prior_x)
        elif x.shape[0] != z.shape[0]:
            # compute action value for one action set
            out = torch.mm(theta, x.T).squeeze(0)
            prior_out = torch.mm(prior_theta, prior_x.T).squeeze(0)
        elif x.shape == theta.shape:
            out = torch.sum(x * theta, -1)
            prior_out = torch.sum(prior_x * prior_theta, -1)
        else:
            # compute predict reward in batch
            out = torch.bmm(theta, x.unsqueeze(-1)).squeeze(-1)
            prior_out = torch.bmm(prior_theta, prior_x.unsqueeze(-1)).squeeze(-1)

        out = self.posterior_scale * out + self.prior_scale * prior_out
        return out

    def regularization(self, z):
        theta = self.hyper_weight(z)
        reg_loss = theta.pow(2).mean()
        return reg_loss

    def get_thetas(self, z):
        theta = self.hyper_weight(z)
        prior_theta = self.prior_weight(z)
        theta = self.posterior_scale * theta + self.prior_scale * prior_theta
        return theta


class HyperNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int] = (),
        noise_dim: int = 2,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        hyper_bias: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        self.basedmodel = mlp(in_features, hidden_sizes)
        self.priormodel = mlp(in_features, hidden_sizes)
        for param in self.priormodel.parameters():
            param.requires_grad = False

        hyper_out_features = in_features if len(hidden_sizes) == 0 else hidden_sizes[-1]
        self.out = HyperLinear(
            noise_dim,
            hyper_out_features,
            prior_scale=prior_scale,
            posterior_scale=posterior_scale,
            use_bias=hyper_bias,
            device=device,
        )
        self.device = device

    def forward(self, z, x):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, device=self.device)
        if isinstance(z, np.ndarray):
            z = torch.as_tensor(z, device=self.device)
        # z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        # x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        logits = self.basedmodel(x)
        prior_logits = self.priormodel(x)
        out = self.out(z, logits, prior_logits)
        return out

    def regularization(self, z):
        # z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        return self.out.regularization(z)
