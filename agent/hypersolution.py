from typing import Sequence
from functools import partial
import sys

sys.path.append("..")

import numpy as np
import torch
import torch.nn.functional as F

from utils import sample_action_noise, sample_update_noise, sample_buffer_noise
from network import HyperNet, EnsembleNet, EpiNet


class ReplayBuffer:
    def __init__(
        self, buffer_size, buffer_shape, noise_type="sp", save_full_feature=False
    ):
        self.buffers = {
            key: np.empty([buffer_size, *shape], dtype=np.float32)
            for key, shape in buffer_shape.items()
        }
        self.buffer_size = buffer_size
        self.noise_dim = buffer_shape["z"][-1]
        self.save_full_feature = save_full_feature
        self.sample_num = 0
        self.set_buffer_noise(noise_type)

    def set_buffer_noise(self, noise_type):
        args = {"M": self.noise_dim}
        if noise_type == "gs":
            self.gen_noise = partial(sample_buffer_noise, "Gaussian", **args)
        elif noise_type == "sp":
            self.gen_noise = partial(sample_buffer_noise, "Sphere", **args)
        elif noise_type == "pn":
            self.gen_noise = partial(sample_buffer_noise, "UnifCube", **args)
        elif noise_type == "pm":
            self.gen_noise = partial(sample_buffer_noise, "PMCoord", **args)
        elif noise_type == "oh":
            self.gen_noise = partial(sample_buffer_noise, "OH", **args)
        elif noise_type == "sps":
            self.gen_noise = partial(sample_buffer_noise, "Sparse", **args)
        elif noise_type == "spc":
            self.gen_noise = partial(sample_buffer_noise, "SparseConsistent", **args)

    def __len__(self):
        return self.sample_num

    def _sample(self, index):
        if self.save_full_feature:
            a_data = self.buffers["a"][: self.sample_num]
            f_data = s_data[np.arange(self.sample_num), a_data.astype(np.int32)][index]
            s_data = self.buffers["s"][: self.sample_num][index]
        else:
            f_data = self.buffers["f"][: self.sample_num][index]
            s_data = None
        r_data = self.buffers["r"][: self.sample_num][index]
        z_data = self.buffers["z"][: self.sample_num][index]
        return s_data, f_data, r_data, z_data

    def reset(self):
        self.sample_num = 0

    def put(self, transition):
        if self.save_full_feature:
            for k, v in transition.items():
                self.buffers[k][self.sample_num] = v
        else:
            self.buffers["r"][self.sample_num] = transition["r"]
            self.buffers["f"][self.sample_num] = transition["s"][transition["a"]]
        z = self.gen_noise()
        self.buffers["z"][self.sample_num] = z
        self.sample_num += 1

    def get(self, shuffle=True):
        # get all data in buffer
        index = list(range(self.sample_num))
        if shuffle:
            np.random.shuffle(index)
        return self._sample(index)

    def sample(self, n):
        # get n data in buffer
        index = np.random.randint(low=0, high=self.sample_num, size=n)
        return self._sample(index)

    def sample_all(self):
        return self._sample(range(self.sample_num))


class HyperSolution:
    def __init__(
        self,
        noise_dim: int,
        n_action: int,
        n_feature: int,
        class_num: int = 1,
        hidden_sizes: Sequence[int] = (),
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        batch_size: int = 32,
        lr: float = 0.01,
        optim: str = "Adam",
        fg_lambda: float = 0.0,
        fg_decay: bool = True,
        based_weight_decay: float = 0.01,
        hyper_weight_decay: float = 0.01,
        noise_coef: float = 0.01,
        buffer_size: int = 10000,
        buffer_noise: str = "sp",
        NpS: int = 20,
        action_noise: str = "sgs",
        update_noise: str = "pn",
        model_type: str = "hyper",
        out_bias: bool = True,
        reset: bool = False,
    ):
        self.noise_dim = noise_dim
        self.action_dim = n_action
        self.feature_dim = n_feature
        self.class_num = class_num
        self.hidden_sizes = hidden_sizes
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.lr = lr
        self.fg_lambda = fg_lambda
        self.fg_decay = fg_decay
        self.batch_size = batch_size
        self.NpS = NpS
        self.optim = optim
        self.based_weight_decay = based_weight_decay
        self.hyper_weight_decay = hyper_weight_decay
        self.noise_coef = noise_coef
        self.buffer_size = buffer_size
        self.action_noise = action_noise
        self.update_noise = update_noise
        self.buffer_noise = buffer_noise
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.out_bias = out_bias

        self.init_model_optimizer()
        self.init_buffer()
        self.set_update_noise()
        self.set_action_noise()
        self.update = (
            getattr(self, "_update_reset") if reset else getattr(self, "_update")
        )

    def init_model_optimizer(self):
        # init hypermodel
        model_param = {
            "in_features": self.feature_dim,
            "hidden_sizes": self.hidden_sizes,
            "noise_dim": self.noise_dim,
            "prior_scale": self.prior_scale,
            "posterior_scale": self.posterior_scale,
            "device": self.device,
        }
        if self.model_type == "hyper":
            Net = HyperNet
            model_param.update({"hyper_bias": self.out_bias})
        elif self.model_type == "epinet":
            model_param.update({"class_num": self.class_num})
            Net = EpiNet
        elif self.model_type == "ensemble":
            Net = EnsembleNet
            model_param.update({"out_bias": self.out_bias})
        else:
            raise NotImplementedError
        self.model = Net(**model_param).to(self.device)
        print(f"\nNetwork structure:\n{str(self.model)}")
        print(
            f"Network parameters: {sum(param.numel() for param in self.model.parameters() if param.requires_grad)}"
        )
        # init optimizer
        trainable_params = [
            {
                "params": (
                    p
                    for name, p in self.model.named_parameters()
                    if "basedmodel" in name and "prior" not in name
                ),
                "weight_decay": self.based_weight_decay,
            },
            {
                "params": (
                    p
                    for name, p in self.model.named_parameters()
                    if "out" in name and "prior" not in name
                ),
                "weight_decay": self.hyper_weight_decay,
            },
        ]
        if self.optim == "Adam":
            self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr)
        elif self.optim == "SGD":
            self.optimizer = torch.optim.SGD(trainable_params, lr=self.lr, momentum=0.9)
        else:
            raise NotImplementedError

    def init_buffer(self):
        # init replay buffer
        # buffer_shape = {
        #     "s": (self.action_dim, self.feature_dim),
        #     "a": (),
        #     "r": (),
        #     "z": (self.noise_dim,),
        # }
        buffer_shape = {"f": (self.feature_dim,), "r": (), "z": (self.noise_dim,)}
        self.buffer = ReplayBuffer(self.buffer_size, buffer_shape, self.buffer_noise)

    def _update(self):
        s_batch, f_batch, r_batch, z_batch = self.buffer.sample(self.batch_size)
        self.learn(s_batch, f_batch, r_batch, z_batch)

    def _update_reset(self):
        sample_num = len(self.buffer)
        if sample_num > self.batch_size:
            s_data, f_data, r_data, z_data = self.buffer.get()
            for i in range(0, self.batch_size, sample_num):
                s_batch, f_batch, r_batch, z_batch = (
                    s_data[i : i + self.batch_size],
                    f_data[i : i + self.batch_size],
                    r_data[i : i + self.batch_size],
                    z_data[i : i + self.batch_size],
                )
                self.learn(s_batch, f_batch, r_batch, z_batch)
            if sample_num % self.batch_size != 0:
                last_sample = sample_num % self.batch_size
                index1 = -np.arange(1, last_sample + 1).astype(np.int32)
                index2 = np.random.randint(
                    low=0, high=sample_num, size=self.batch_size - last_sample
                )
                index = np.hstack([index1, index2])
                s_batch, f_batch, r_batch, z_batch = (
                    s_data[index],
                    f_data[index],
                    r_data[index],
                    z_data[index],
                )
                self.learn(s_batch, f_batch, r_batch, z_batch)
        else:
            s_batch, f_batch, r_batch, z_batch = self.buffer.sample(self.batch_size)
            self.learn(s_batch, f_batch, r_batch, z_batch)

    def put(self, transition):
        self.buffer.put(transition)

    def learn(self, s_batch, f_batch, r_batch, z_batch):
        z_batch = torch.FloatTensor(z_batch).to(self.device)
        f_batch = torch.FloatTensor(f_batch).to(self.device)
        r_batch = torch.FloatTensor(r_batch).to(self.device)
        if s_batch is not None:
            s_batch = torch.FloatTensor(s_batch).to(self.device)

        # noise for update
        update_noise = torch.from_numpy(self.gen_update_noise()).to(self.device)
        # noise for target
        target_noise = torch.bmm(update_noise, z_batch.unsqueeze(-1)) * self.noise_coef

        predict = self.model(update_noise, f_batch)
        if self.model_type == "epinet" and self.class_num > 1:
            r_batch = r_batch.unsqueeze(-1).repeat(1, self.NpS).to(torch.int64)
            r_batch = r_batch.view(-1)
            predict = predict.view(-1, self.class_num)
            loss = F.cross_entropy(predict, r_batch)
        else:
            diff = target_noise.squeeze(-1) + r_batch.unsqueeze(-1) - predict
            diff = diff.pow(2).mean(-1)
            if self.fg_lambda:
                fg_lambda = (
                    self.fg_lambda / np.sqrt(len(self.buffer))
                    if self.fg_decay
                    else self.fg_lambda
                )
                fg_term = self.model(update_noise, s_batch)
                fg_term = fg_term.max(dim=-1)[0]
                loss = (diff - fg_lambda * fg_term).mean()
            else:
                loss = diff.mean()

        for param_group in self.optimizer.param_groups:
            param_group["weight_decay"] = self.hyper_weight_decay / len(self.buffer)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_thetas(self, num=1):
        assert len(self.hidden_sizes) == 0, f"hidden size > 0"
        action_noise = self.gen_action_noise(dim=num)
        with torch.no_grad():
            thetas = self.model.out.get_thetas(action_noise).cpu().numpy()
        return thetas

    def predict(self, features, num=1):
        action_noise = self.gen_action_noise(dim=num)
        with torch.no_grad():
            p_a = self.model(action_noise, features).cpu().numpy()
        return p_a

    def set_action_noise(self):
        args = {"M": self.noise_dim}
        if self.action_noise == "gs":
            self.gen_action_noise = partial(sample_action_noise, "Gaussian", **args)
        elif self.action_noise == "sp":
            self.gen_action_noise = partial(sample_action_noise, "Sphere", **args)
        elif self.action_noise == "pn":
            self.gen_action_noise = partial(sample_action_noise, "UnifCube", **args)
        elif self.action_noise == "pm":
            self.gen_action_noise = partial(sample_action_noise, "PMCoord", **args)
        elif self.action_noise == "oh":
            self.gen_action_noise = partial(sample_action_noise, "OH", **args)
        elif self.action_noise == "sps":
            self.gen_action_noise = partial(sample_action_noise, "Sparse", **args)
        elif self.action_noise == "spc":
            self.gen_action_noise = partial(
                sample_action_noise, "SparseConsistent", **args
            )

    def set_update_noise(self):
        args = {"M": self.noise_dim, "dim": self.NpS, "batch_size": self.batch_size}
        if self.update_noise == "gs":
            self.gen_update_noise = partial(sample_update_noise, "Gaussian", **args)
        elif self.update_noise == "sp":
            self.gen_update_noise = partial(sample_update_noise, "Sphere", **args)
        elif self.update_noise == "pn":
            self.gen_update_noise = partial(sample_update_noise, "UnifCube", **args)
        elif self.update_noise == "pm":
            self.gen_update_noise = partial(sample_update_noise, "PMCoord", **args)
        elif self.update_noise == "oh":
            self.gen_update_noise = partial(sample_update_noise, "OH", **args)
        elif self.update_noise == "sps":
            self.gen_update_noise = partial(sample_update_noise, "Sparse", **args)
        elif self.update_noise == "spc":
            self.gen_update_noise = partial(
                sample_update_noise, "SparseConsistent", **args
            )

    def reset(self):
        self.init_model_optimizer()
