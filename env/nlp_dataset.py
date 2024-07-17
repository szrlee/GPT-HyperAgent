import numpy as np

from itertools import cycle
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


class HateSpeechDataset(Dataset):
    def __init__(self, dataset="hatespeech", llm_name="gpt2", max_length=2024):
        dataset_path = f"/apdcephfs/share_1563664/ztjiaweixu/huggingface/{dataset}"
        self.dataset = load_from_disk(dataset_path)["train"]
        tokenizer_path = (
            f"/apdcephfs/share_1563664/ztjiaweixu/huggingface/{llm_name}/tokenizer"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        score = item["hate_speech_score"]
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        input_ids, attention_mask = (
            inputs["input_ids"].squeeze(),
            inputs["attention_mask"].squeeze(),
        )
        # Score is passed to simulate rewards during the learning loop
        reward = np.ones(2) * 0.5
        reward[1] = 1 if score <= 0.5 else -0.5
        action = 1 if score <= 0.5 else 0
        return input_ids, attention_mask, score, reward, action


class HateSpeechEnv:
    def __init__(
        self,
        n_features=1024,
        n_actions=32,
        eta=0.1,
        sigma=1,
        llm_name="gpt2",
    ):
        reward_random_state = np.random.randint(1, 312414)
        self.reward_random = np.random.default_rng(reward_random_state)

        dataset = HateSpeechDataset(llm_name=llm_name, max_length=n_features)
        self.all_actions = len(dataset)

        self.n_actions = n_actions
        self.n_features = n_features
        self.sub_actions = n_actions

        self.dataloader = DataLoader(dataset, batch_size=n_actions, shuffle=True, drop_last=True)
        self.dataloader = cycle(iter(self.dataloader))

        self.eta = eta
        self.alg_prior_sigma = sigma

        self.set_context()

    def set_context(self):
        data = next(self.dataloader)
        input_ids, attention_mask, score, reward, true_action = data
        self.features = (input_ids, attention_mask)
        self.score = score
        self.sub_reward = reward
        self.true_action = true_action

    def get_feature(self):
        input_ids, attention_mask = self.features
        return input_ids, attention_mask

    def reward(self, arm):
        reward = self.sub_reward[np.arange(self.n_actions), arm]
        return reward

    def regret(self, arm):
        expect_reward = self.score[arm]
        best_arm_reward = self.score.max()
        return best_arm_reward - expect_reward

    def expect_regret(self, arm, features):
        """
        Compute the regret of a single step
        """
        expect_reward = self.sub_reward[np.arange(self.n_actions), arm]
        best_arm_reward = self.sub_reward.max(dim=1).values
        return best_arm_reward - expect_reward
