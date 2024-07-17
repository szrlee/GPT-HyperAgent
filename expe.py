""" Packages import """

from env.synthetic import SyntheticNonlinModel
from env.nlp_dataset import HateSpeechEnv
from utils import (
    plotRegret,
    storeRegret,
)
import numpy as np


def MAB_expe(
    n_expe,
    n_features,
    n_arms,
    T,
    methods,
    param_dic,
    labels,
    colors,
    path,
    problem="Quadratic",
    doplot=False,
    freq_task=True,
    seed=2022,
    **kwargs,
):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with n_arms Linear Gaussian Bandits with multivariate Gaussian prior
    :param n_expe: int, number of experiments
    :param n_features: int, dimension of feature vectors
    :param n_arms: int, number of arms
    :param T: int, time horizon
    :param methods: list, algorithms to use
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param doplot: boolean, plot the curves or not
    :param problem: str, choose from {'FreqRusso', 'Zhang', 'Russo', 'movieLens'}
    :param freq_task: boolean, Freq MOD for task
    :param path: str
    :return: dict, regrets, quantiles, means, stds of final regrets for each methods
    """
    from agent.hyper import HyperMAB
    if problem == "Quadratic":
        reward_version = "v1"
    elif problem == "Neural":
        reward_version = "v2"

    models = [
        HyperMAB(
            SyntheticNonlinModel(
                n_features,
                n_arms,
                all_actions=kwargs["all_arms"],
                reward_version=reward_version,
                freq_task=freq_task,
                eta=kwargs.get("eta", 0.1),
            )
        )
        for _ in range(n_expe)
    ]
    title = f"Nonlinear Bandit Model  - n_arms: {n_arms} - n_features: {n_features} - reward: {reward_version}"

    print("Begin experiments on '{}'".format(title))
    results = storeRegret(
        models, methods, param_dic, n_expe, T, path, seed, use_torch=True
    )
    if doplot:
        plotRegret(labels, results, colors, title, path, log=False)
    return results

def Textual_expe(
    n_expe,
    n_features,
    n_arms,
    T,
    methods,
    param_dic,
    labels,
    colors,
    path,
    problem="hatespeech",
    llm_name="gpt2",
    doplot=False,
    seed=2022,
    **kwargs,
):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with n_arms Linear Gaussian Bandits with multivariate Gaussian prior
    :param n_expe: int, number of experiments
    :param n_features: int, dimension of feature vectors
    :param n_arms: int, number of arms
    :param T: int, time horizon
    :param methods: list, algorithms to use
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param doplot: boolean, plot the curves or not
    :param problem: str, choose from {'FreqRusso', 'Zhang', 'Russo', 'movieLens'}
    :param path: str
    :return: dict, regrets, quantiles, means, stds of final regrets for each methods
    """
    from agent.hyper import HyperMAB

    if problem == "hatespeech":
        models = [HyperMAB(HateSpeechEnv(n_features, n_arms, llm_name=llm_name)) for _ in range(n_expe)]
        title = f"HateSpeech  - n_arms: {n_arms} - n_features: {n_features}"
    else:
        raise NotImplementedError

    print("Begin experiments on '{}'".format(title))
    results = storeRegret(
        models, methods, param_dic, n_expe, T, path, seed, use_torch=True
    )
    if doplot:
        plotRegret(labels, results, colors, title, path, log=False)
    return results
