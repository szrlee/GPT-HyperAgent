""" Packages import """

import os
import csv
import inspect
import numpy as np
import random as rd
import itertools as it
import scipy.stats as st
import matplotlib.pyplot as plt

from tqdm import tqdm
from logger import configure

cmap = {
    0: "black",
    1: "blue",
    2: "yellow",
    3: "green",
    4: "red",
    5: "grey",
    6: "purple",
    7: "brown",
    8: "pink",
    9: "cyan",
}


mapping_methods_labels = {
    "Hyper": "Hyper",
    "EpiNet": "EpiNet",
    "Ensemble": "Ensemble",
}


mapping_methods_colors = {
    "Hyper": "blue",
    "EpiNet": "green",
    "Ensemble": "red",
}


def labelColor(methods):
    """
    Map methods to labels and colors for regret curves
    :param methods: list, list of methods
    :return: lists, labels and vectors
    """
    labels = [
        mapping_methods_labels[m] if m in mapping_methods_labels.keys() else m
        for m in methods
    ]
    colors = [
        mapping_methods_colors[m] if m in mapping_methods_colors.keys() else None
        for m in methods
    ]
    return labels, colors


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return rd.choice(indices)


def rd_max(vector):
    """
    Compute random among eligible maximum value
    :param vector: np.array
    :return: int, random index among eligible maximum value
    """
    index = rd_argmax(vector)
    return vector[index]


def haar_matrix(M):
    """
    Haar random matrix generation
    """
    z = np.empty((M, M), dtype=np.float32)
    z[:] = np.random.randn(M, M)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return np.multiply(q, ph)


def sphere_matrix(N, M):
    v = np.empty((N, M), dtype=np.float32)
    v[:] = np.random.randn(N, M)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def normal_matrix(N, M):
    v = np.empty((N, M), dtype=np.float32)
    v[:] = np.random.randn(N, M)
    return v


def random_choice_noreplace(m, n, axis=-1):
    # m, n are the number of rows, cols of output
    return np.random.rand(m, n).argsort(axis=axis)


def sample_buffer_noise(noise_type, M, sparsity=2):
    # ensure the sampled vector is isotropic
    assert M > 0
    if noise_type == "Sphere":
        return sphere_matrix(1, M)[0]
    elif noise_type == "Gaussian" or noise_type == "Normal":
        return normal_matrix(1, M)[0] / np.sqrt(M)
    elif noise_type == "PMCoord":
        i = np.random.choice(M)
        B = np.zeros(M, dtype=np.float32)
        B[i] = random_sign()
        return B
    elif noise_type == "OH":
        i = np.random.randint(0, M)
        B = np.zeros(M, dtype=np.float32)
        B[i] = 1
        return B
    elif noise_type == "Sparse":
        i = random_choice_noreplace(1, M)[0, :sparsity]
        B = np.zeros(M, dtype=np.float32)
        B[i] = random_sign(1 * sparsity)
        return B / np.sqrt(sparsity)
    elif noise_type == "SparseConsistent":
        i = random_choice_noreplace(1, M)[0, :sparsity]
        B = np.zeros(M, dtype=np.float32)
        B[i] = random_sign(1)
        return B / np.sqrt(sparsity)
    elif noise_type == "UnifCube":
        B = np.empty(M, dtype=np.float32)
        B[:] = 2 * np.random.binomial(1, 0.5, M) - 1
        return B / np.sqrt(M)
    else:
        raise NotImplementedError


def sample_action_noise(noise_type, M, dim=1, sparsity=2):
    # ensure the sampled vector is isotropic
    assert M > 0
    if noise_type == "Sphere":
        return sphere_matrix(dim, M) * np.sqrt(M)
    elif noise_type == "Gaussian" or noise_type == "Normal":
        return normal_matrix(dim, M)
    elif noise_type == "PMCoord":
        i = np.random.choice(M, dim)
        B = np.zeros((dim, M), dtype=np.float32)
        B[np.arange(dim), i] = random_sign(dim)
        return B * np.sqrt(M)
    elif noise_type == "OH":
        i = np.random.randint(0, M, dim)
        B = np.zeros((dim, M), dtype=np.float32)
        B[np.arange(dim), i] = 1
        return B
    elif noise_type == "Sparse":
        i = random_choice_noreplace(dim, M)[:, :sparsity]
        B = np.zeros((dim, M), dtype=np.float32)
        B[np.expand_dims(np.arange(dim), axis=1), i] = random_sign(
            dim * sparsity
        ).reshape(dim, sparsity)
        return B / np.sqrt(sparsity) * np.sqrt(M)
    elif noise_type == "SparseConsistent":
        i = random_choice_noreplace(dim, M)[:, :sparsity]
        B = np.zeros((dim, M), dtype=np.float32)
        B[np.expand_dims(np.arange(dim), axis=1), i] = random_sign(dim).reshape(dim, 1)
        return B / np.sqrt(sparsity) * np.sqrt(M)
    elif noise_type == "UnifCube":
        B = np.empty((dim, M), dtype=np.float32)
        B[:] = 2 * np.random.binomial(1, 0.5, (dim, M)) - 1
        return B
    else:
        raise NotImplementedError


def sample_update_noise(noise_type, M, dim=1, sparsity=2, batch_size=1):
    # ensure the sampled vector is isotropic
    assert M > 0
    if noise_type == "Sphere":
        v = np.empty((batch_size, dim, M), dtype=np.float32)
        v[:] = np.random.randn(batch_size, dim, M)
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        return v * np.sqrt(M)
    elif noise_type == "Gaussian" or noise_type == "Normal":
        v = np.empty((batch_size, dim, M), dtype=np.float32)
        v[:] = np.random.randn(batch_size, dim, M)
        return v
    elif noise_type == "PMCoord":
        B = np.zeros((M * 2, M), dtype=np.float32)
        B[np.arange(M), np.arange(M)] = 1
        B[np.arange(M) + M, np.arange(M)] = -1
        B = np.expand_dims(B, 0).repeat(batch_size, 0)
        return B * np.sqrt(M)
    elif noise_type == "OH":
        B = np.eye(M, dtype=np.float32)
        B = np.expand_dims(B, 0).repeat(batch_size, 0)
        return B
    elif noise_type == "Sparse":
        index = np.array([list(c) for c in it.combinations(list(range(M)), sparsity)])
        elements = list(it.product([1, -1], repeat=sparsity))
        n = len(index)
        B = []
        for e in elements:
            e = np.expand_dims(np.array(e), 0).repeat(n, 0)
            b = np.zeros((n, M), dtype=np.float32)
            b[np.expand_dims(np.arange(n), axis=1), index] = e
            B.append(b)
        B = np.concatenate(B, 0)
        B = np.expand_dims(B, 0).repeat(batch_size, 0)
        return B / np.sqrt(sparsity) * np.sqrt(M)
    elif noise_type == "SparseConsistent":
        index = np.array([list(c) for c in it.combinations(list(range(M)), sparsity)])
        n = len(index)
        B = []
        for element in [1, -1]:
            e = np.ones((n, sparsity)) * element
            b = np.zeros((n, M), dtype=np.float32)
            b[np.expand_dims(np.arange(n), axis=1), index] = e
            B.append(b)
        B = np.concatenate(B, 0)
        B = np.expand_dims(B, 0).repeat(batch_size, 0)
        return B / np.sqrt(sparsity) * np.sqrt(M)
    elif noise_type == "UnifCube":
        B = np.array(list((it.product(range(2), repeat=M))), dtype=np.float32)
        B = B * 2 - 1
        B = np.expand_dims(B, 0).repeat(batch_size, 0)
        return B
    else:
        raise NotImplementedError


def multi_haar_matrix(N, M):
    v = np.zeros(((N // M + 1) * M, M))
    for _ in range(N // M + 1):
        v[np.arange(M), :] = haar_matrix(M)
    return v[np.arange(N), :]


def display_results(delta, g, ratio, p_star):
    """
    Display quantities of interest in IDS algorithm
    """
    print("delta {}".format(delta))
    print("g {}".format(g))
    print("ratio : {}".format(ratio))
    print("p_star {}".format(p_star))


def plotRegret(labels, regret, colors, title, path, log=False):
    """
    Plot Bayesian regret
    :param labels: list, list of labels for the different curves
    :param mean_regret: np.array, averaged regrets from t=1 to T for all methods
    :param colors: list, list of colors for the different curves
    :param title: string, plot's title
    """

    all_regrets = regret["all_regrets"]
    mean_regret = regret["mean_regret"]
    plt.figure(figsize=(10, 8), dpi=80)

    T = mean_regret.shape[1]
    for i, l in enumerate(labels):
        c = cmap[i] if not colors else colors[i]
        x = np.arange(T)
        low_CI_bound, high_CI_bound = st.t.interval(
            0.95, T - 1, loc=mean_regret[i], scale=st.sem(all_regrets[i])
        )
        # low_CI_bound = np.quantile(all_regrets[i], 0.05, axis=0)
        # high_CI_bound = np.quantile(all_regrets[i], 0.95, axis=0)
        plt.plot(x, mean_regret[i], c=c, label=l)
        plt.fill_between(x, low_CI_bound, high_CI_bound, color=c, alpha=0.2)
        if log:
            plt.yscale("log")
    plt.grid(color="grey", linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.ylabel("Cumulative regret")
    plt.xlabel("Time period")
    plt.legend(loc="best")
    plt.savefig(path + "/regret.pdf")


def storeRegret(
    models, methods, param_dic, n_expe, T, path, seed=2022, use_torch=False
):
    """
    Compute the experiment for all specified models and methods
    :param models: list of MAB
    :param methods: list of algorithms
    :param param_dic: parameters for all the methods
    :param n_expe: number of trials
    :param T: Time horizon
    :return: Dictionnary with results from the experiments
    """
    all_regrets = np.zeros((len(methods), n_expe, T), dtype=np.float32)
    for i, m in enumerate(methods):
        set_seed(seed, use_torch=use_torch)
        alg_name = m.split(":")[0]
        logger = configure(path, ["csv"])
        for j in tqdm(range(n_expe)):
            model = models[j]
            alg = model.__getattribute__(alg_name)
            args = inspect.getfullargspec(alg)[0][3:]
            args = [T, logger] + [param_dic[m][i] for i in args]
            reward, regret = alg(*args)
            all_regrets[i, j, :] = np.cumsum(regret)
        print(f"{m}: {np.mean(all_regrets[i], axis=0)[-1]}")

    mean_regret = all_regrets.mean(axis=1)
    results = {
        "mean_regret": mean_regret,
        "all_regrets": all_regrets,
    }
    return results

def random_sign(N=None):
    if (N is None) or (N == 1):
        return np.random.randint(0, 2, 1) * 2 - 1
    elif N > 1:
        return np.random.randint(0, 2, N) * 2 - 1


def set_seed(seed, use_torch=False):
    np.random.seed(seed)
    rd.seed(seed)
    if use_torch:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(seed)
