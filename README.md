# Usage

To run the experiment on Neural Bandit, use the following command:
```
python -m scripts.run_hyper --game Neural --method Hyper
```

To run the experiment on Online Automated Content Moderation, use this command:
```
python -m scripts.run_llm --game hatespeech --model-type=linear --llm-name=gpt2
```

HyperAgent for deep reinforcement learning can be found in the repo [szrlee/HyperAgent](https://github.com/szrlee/HyperAgent).

## Citation

If you find this work useful to your research, please cite our paper.

```bibtex
@misc{li2024scaling,
  author = {Li, Yingru and Xu, Jiawei and Luo, Zhi-Quan},
  title  = {{Adaptive Foundation Models for Online Decisions: HyperAgent with Fast Incremental Uncertainty Estimation}},
  howpublished  = {Preprint},
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
