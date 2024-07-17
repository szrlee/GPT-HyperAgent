from functools import partial
from pprint import pprint
import sys

sys.path.append("..")

import numpy as np
import torch
import torch.nn.functional as F

from utils import sample_action_noise, sample_update_noise

from network import HyperLLM


def _random_argmax(vals, scale=1e-7):
    """Select max with additional random noise."""
    noise = torch.distributions.Uniform(0, 1).sample(vals.shape).to(vals.device)
    index = torch.max(vals + scale * noise, dim=-1)[1]
    return index


class ReplayBuffer:
    def __init__(self, buffer_size, buffer_shape, noise_type="sp"):
        self.buffers = {
            key: np.empty([buffer_size, *shape], dtype=np.float32)
            for key, shape in buffer_shape.items()
        }
        self.buffer_size = buffer_size
        self.noise_dim = buffer_shape["z"][-1]
        self.current_size = 0
        self.point = 0
        self.set_buffer_noise(noise_type)

    def set_buffer_noise(self, noise_type):
        args = {"M": self.noise_dim}
        if noise_type == "gs":
            self.gen_noise = partial(sample_action_noise, "Gaussian", **args)
        elif noise_type == "sp":
            self.gen_noise = partial(sample_action_noise, "Sphere", **args)
        elif noise_type == "pn":
            self.gen_noise = partial(sample_action_noise, "UnifCube", **args)
        elif noise_type == "pm":
            self.gen_noise = partial(sample_action_noise, "PMCoord", **args)
        elif noise_type == "oh":
            self.gen_noise = partial(sample_action_noise, "OH", **args)
        elif noise_type == "sps":
            self.gen_noise = partial(sample_action_noise, "Sparse", **args)
        elif noise_type == "spc":
            self.gen_noise = partial(sample_action_noise, "SparseConsistent", **args)

    def __len__(self):
        return self.current_size

    def _sample(self, index):
        input_ids = self.buffers["input_ids"][index]
        attention_mask = self.buffers["attention_mask"][index]
        a_data = self.buffers["a"][index]
        r_data = self.buffers["r"][index]
        z_data = self.buffers["z"][index]
        return input_ids, attention_mask, a_data, r_data, z_data

    def reset(self):
        self.current_size = 0

    def put(self, transition):
        batch_size = len(transition["r"])
        idx = self._get_ordered_storage_idx(batch_size)
        for k, v in transition.items():
            self.buffers[k][idx] = v
        z = self.gen_noise(dim=batch_size) / np.sqrt(self.noise_dim)
        self.buffers["z"][idx] = z

    def get(self, shuffle=True):
        # get all data in buffer
        index = list(range(self.current_size))
        if shuffle:
            np.random.shuffle(index)
        return self._sample(index)

    def sample(self, n):
        # get n data in buffer
        index = np.random.randint(low=0, high=self.current_size, size=n)
        return self._sample(index)

    def sample_all(self):
        return self._sample(range(self.current_size))

    # if full, insert in order
    def _get_ordered_storage_idx(self, inc=None):
        inc = inc or 1  # size increment
        assert inc <= self.buffer_size, "Batch committed to replay is too large!"

        if self.point + inc <= self.buffer_size - 1:
            idx = np.arange(self.point, self.point + inc)
        else:
            overflow = inc - (self.buffer_size - self.point)
            idx_a = np.arange(self.point, self.buffer_size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])

        self.point = (self.point + inc) % self.buffer_size

        # update replay size, don't add when it already surpass self.size
        if self.current_size < self.buffer_size:
            self.current_size = min(self.buffer_size, self.current_size + inc)

        if inc == 1:
            idx = idx[0]
        return idx


class HyperLLMSolution:
    def __init__(
        self,
        noise_dim: int,
        n_action: int,
        n_feature: int,
        action_num: int = 2,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        batch_size: int = 32,
        lr: float = 0.01,
        optim: str = "Adam",
        based_weight_decay: float = 0.01,
        hyper_weight_decay: float = 0.01,
        noise_coef: float = 0.01,
        buffer_size: int = 10000,
        buffer_noise: str = "sp",
        NpS: int = 20,
        action_noise: str = "sgs",
        update_noise: str = "pn",
        model_type: str = "hyper",
        llm_name: str = "gpt2",
        fine_tune: bool = False,
        out_bias: bool = True,
    ):
        self.noise_dim = noise_dim
        self.n_action = n_action
        self.n_feature = n_feature
        self.action_num = action_num
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.lr = lr
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
        self.model_type = model_type
        self.llm_name = llm_name
        self.fine_tune = fine_tune
        self.out_bias = out_bias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.init_model_optimizer()
        self.init_buffer()
        self.set_update_noise()
        self.set_action_noise()

    def init_model_optimizer(self):
        # init model
        model_param = {
            "noise_dim": self.noise_dim,
            "action_num": self.action_num,
            "prior_scale": self.prior_scale,
            "posterior_scale": self.posterior_scale,
            "head_name": self.model_type,
            "llm_name": self.llm_name,
            "fine_tune": self.fine_tune,
            "out_bias": self.out_bias,
            "device": self.device,
        }
        self.model = HyperLLM(**model_param).to(self.device)
        param_dict = {"Trainable": [], "Frozen": []}
        trainable_param_size, frozen_param_size = 0, 0
        for name, param in self.model.named_parameters():
            # if "transformer" not in name: continue
            if param.requires_grad:
                trainable_param_size += param.numel()
                param_dict["Trainable"].append(name)
            else:
                frozen_param_size += param.numel()
                param_dict["Frozen"].append(name)
        pprint(param_dict)
        print(f"\nNetwork structure:\n{str(self.model)}")
        print(
            f"Network parameters: Trainable {trainable_param_size}, Frozen {frozen_param_size}"
        )
        # init optimizer
        trainable_params = [
            {
                "params": (
                    p
                    for name, p in self.model.named_parameters()
                    if "transformer" in name and p.requires_grad
                ),
                "weight_decay": self.based_weight_decay,
            },
            {
                "params": (
                    p
                    for name, p in self.model.named_parameters()
                    if "out" in name and p.requires_grad
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
        buffer_shape = {
            "input_ids": (self.n_feature,),
            "attention_mask": (self.n_feature,),
            "a": {},
            "r": (),
            "z": (self.noise_dim,),
        }
        self.buffer = ReplayBuffer(self.buffer_size, buffer_shape, self.buffer_noise)

    def update(self):
        input_ids, attention_mask, a_batch, r_batch, z_batch = self.buffer.sample(
            self.batch_size
        )
        results = self.learn(input_ids, attention_mask, a_batch, r_batch, z_batch)
        return results

    def put(self, transition):
        self.buffer.put(transition)

    def learn(self, input_ids, attention_mask, a_batch, r_batch, z_batch):
        z_batch = torch.FloatTensor(z_batch).to(self.device)
        r_batch = torch.FloatTensor(r_batch).to(self.device)
        a_batch = torch.FloatTensor(a_batch).to(dtype=torch.int64, device=self.device)
        input_ids = torch.FloatTensor(input_ids).to(
            dtype=torch.int64, device=self.device
        )
        attention_mask = torch.from_numpy(attention_mask).to(
            dtype=torch.int64, device=self.device
        )

        # noise for update
        update_noise = torch.from_numpy(self.gen_update_noise()).to(self.device)
        # noise for target
        target_noise = torch.bmm(update_noise, z_batch.unsqueeze(-1)) * self.noise_coef

        predict = self.model(update_noise, input_ids, attention_mask)
        if self.model_type == "linear":
            predict = predict[np.arange(self.batch_size), a_batch]
            target = r_batch
        else:
            a_one_hot = F.one_hot(a_batch, self.action_num).to(
                torch.float32
            )  # (None, n_a)
            predict = torch.einsum("bka,ba->bk", predict, a_one_hot)  # (None, NpS)
            target = target_noise.squeeze(-1) + r_batch.unsqueeze(-1)
        diff = (target - predict).pow(2).mean(-1)
        loss = diff.mean()

        for param_group in self.optimizer.param_groups:
            param_group["weight_decay"] = self.hyper_weight_decay / len(self.buffer)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        results = {"loss": loss.item()}
        return results

    def predict(self, input_ids, attention_mask, num=1):
        action_noise = self.gen_action_noise(dim=num)
        with torch.no_grad():
            p_a = self.model(action_noise, input_ids, attention_mask)  # .cpu().numpy()
            a = _random_argmax(p_a)
        return a.cpu().numpy()

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
