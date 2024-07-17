from typing import List

import numpy as np
import math
import torch
from torch.optim import Optimizer

from network import LinearNet
from .hypersolution import HyperSolution


def lmc(
    params: List[torch.Tensor],
    d_p_list: List[torch.Tensor],
    weight_decay: float,
    lr: float,
):
    r"""Functional API that performs Langevine MC algorithm computation."""

    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add_(param, alpha=weight_decay)

        param.add_(d_p, alpha=-lr)


class LangevinMC(Optimizer):
    def __init__(
        self,
        params,  # parameters of the model
        lr=0.01,  # learning rate
        beta_inv=0.01,  # inverse temperature parameter
        sigma=1.0,  # variance of the Gaussian noise
        weight_decay=1.0,
        device=None,
    ):  # l2 penalty
        if lr < 0:
            raise ValueError("lr must be positive")
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta_inv = beta_inv
        self.lr = lr
        self.sigma = sigma
        self.temp = -math.sqrt(2 * beta_inv / lr) * sigma
        self.curr_step = 0
        defaults = dict(weight_decay=weight_decay)
        super(LangevinMC, self).__init__(params, defaults)

    def init_map(self):
        self.mapping = dict()
        index = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    num_param = p.numel()
                    self.mapping[p] = [index, num_param]
                    index += num_param
        self.total_size = index

    @torch.no_grad()
    def step(self):
        self.curr_step += 1
        if self.curr_step == 1:
            self.init_map()

        lr = self.lr
        temp = self.temp
        noise = temp * torch.randn(self.total_size, device=self.device)

        for group in self.param_groups:
            weight_decay = group["weight_decay"]

            params_with_grad = []
            d_p_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)

                    start, length = self.mapping[p]
                    add_noise = noise[start : start + length].reshape(p.shape)
                    delta_p = p.grad
                    delta_p = delta_p.add_(add_noise)
                    d_p_list.append(delta_p)
                    # p.add_(delta_p)
            lmc(params_with_grad, d_p_list, weight_decay, lr)


class LMCTS(HyperSolution):

    def init_model_optimizer(self):
        # init hypermodel
        model_param = {
            "in_features": self.feature_dim,
            "hidden_sizes": self.hidden_sizes,
            "prior_scale": self.prior_scale,
            "posterior_scale": self.posterior_scale,
            "device": self.device,
        }

        self.model = LinearNet(**model_param).to(self.device)
        print(f"\nNetwork structure:\n{str(self.model)}")
        print(
            f"Network parameters: {sum(param.numel() for param in self.model.parameters() if param.requires_grad)}"
        )
        # init optimizer
        beta_inv = 1e-7 * self.feature_dim * np.log(self.buffer_size)
        self.optimizer = LangevinMC(
            self.model.parameters(),
            lr=self.lr,
            beta_inv=beta_inv,
            weight_decay=self.based_weight_decay,
        )

    def put(self, transition):
        self.buffer.put(transition)

    def _update(self):
        if self.batch_size == 0:
            s_batch, f_batch, r_batch, z_batch = self.buffer.sample_all()
        else:
            s_batch, f_batch, r_batch, z_batch = self.buffer.sample(self.batch_size)
        self.learn(s_batch, f_batch, r_batch, z_batch)

    def learn(self, s_batch, f_batch, r_batch, z_batch):
        # z_batch = torch.FloatTensor(z_batch).to(self.device)
        f_batch = torch.FloatTensor(f_batch).to(self.device)
        r_batch = torch.FloatTensor(r_batch).to(self.device)
        if s_batch is not None:
            s_batch = torch.FloatTensor(s_batch).to(self.device)

        predict = self.model(f_batch)
        diff = r_batch.unsqueeze(-1) - predict
        diff = diff.pow(2).mean(-1)
        if self.fg_lambda:
            fg_lambda = (
                self.fg_lambda / np.sqrt(len(self.buffer))
                if self.fg_decay
                else self.fg_lambda
            )
            fg_term = self.model(s_batch)
            fg_term = fg_term.max(dim=-1)[0]
            loss = (diff - fg_lambda * fg_term).mean()
        else:
            loss = diff.mean()
        # norm_coef = self.norm_coef / len(self.buffer)
        # reg_loss = self.model.regularization(update_noise) * norm_coef
        # loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, features, num=1):
        with torch.no_grad():
            p_a = self.model(features).cpu().numpy()
        return p_a
