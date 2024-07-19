# HyperAgent [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fszrlee%2FHyperAgent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

Author: [Yingru Li](https://richardli.xyz), [Jiawei Xu](https://github.com/jiawei415), [Zhi-Quan Luo](https://en.wikipedia.org/wiki/Zhi-Quan_Tom_Luo)

Welcome to the official implementation of **GPT-HyperAgent**, a cutting-edge algorithm designed for adaptive foundation models in online decision-making. This repository accompanies our paper [Adaptive Foundation Models for Online Decisions: HyperAgent with Fast Incremental Uncertainty Estimation](https://arxiv.org/abs/2407.13195).

GPT-HyperAgent aims to address the challenges of scalable exploration and fast incremental uncertainty estimation in large-scale online decision environments.

## Key Features

- **Fast Incremental Uncertainty Estimation:** Ensures quick updates and reliable uncertainty quantification with logarithmic per-step computational complexity.
- **Scalable Exploration:** Efficiently handles large state-action spaces, facilitating robust and adaptive exploration while matching the regret order of exact Thompson sampling.
- **Integration with GPT Models:** Utilizes the strengths of GPT architectures to enhance decision-making processes in contextual bandits with natural language input.

## Getting Started

To get started with HyperAgent, refer to the detailed documentation and examples provided in this repository. For large-scale deep RL benchmarking results and details, visit the [szrlee/HyperAgent](https://github.com/szrlee/HyperAgent) repository.

We welcome contributions and feedback from the community to help improve and expand the capabilities of HyperAgent.

- To run the experiment on Neural Bandit, use the following command:
```
python -m scripts.run_hyper --game Neural --method Hyper
```

- To run the experiment on Online Automated Content Moderation, use this command:
```
python -m scripts.run_llm --game hatespeech --model-type=linear --llm-name=gpt2
```

HyperAgent for deep reinforcement learning can be found in the repo [szrlee/HyperAgent](https://github.com/szrlee/HyperAgent).

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@misc{li2024onlinefoundationagent,
      title={Adaptive Foundation Models for Online Decisions: HyperAgent with Fast Incremental Uncertainty Estimation}, 
      author={Yingru Li and Jiawei Xu and Zhi-Quan Luo},
      year={2024},
      eprint={2407.13195},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.13195},
      note  = {Presentation at ICML 2024 Workshops: (1) "Aligning Reinforcement Learning Experimentalists and Theorists"; (2) "Automated Reinforcement Learning: Exploring Meta-Learning, AutoML, and LLMs"},
}
```

```bibtex
@inproceedings{li2024hyperagent,
  title         = {{Q-Star Meets Scalable Posterior Sampling: Bridging Theory and Practice via HyperAgent}},
  author        = {Li, Yingru and Xu, Jiawei and Han, Lei and Luo, Zhi-Quan},
  booktitle     = {Forty-first International Conference on Machine Learning},
  year          = {2024},
  series        = {Proceedings of Machine Learning Research},
  eprint        = {2402.10228},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}，
  url           = {https://arxiv.org/abs/2402.10228}
}
```
