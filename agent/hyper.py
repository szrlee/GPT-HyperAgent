import numpy as np

from tqdm import tqdm
from utils import rd_argmax
from agent.hypersolution import HyperSolution
from agent.hyperllmsolution import HyperLLMSolution
from agent.lmcts import LMCTS


class HyperMAB:
    def __init__(self, env):
        self.env = env
        self.expect_regret, self.n_a, self.d, self.features = (
            env.expect_regret,
            env.n_actions,
            env.n_features,
            env.features,
        )
        self.reward, self.eta = env.reward, env.eta

    def set_context(self):
        self.env.set_context()
        self.features = self.env.features

    def LLM(
        self,
        T,
        logger,
        log_interval=10,
        noise_dim=2,
        lr=0.01,
        based_weight_decay=0.0,
        hyper_weight_decay=0.0,
        z_coef=None,
        batch_size=32,
        prior_scale=1.0,
        posterior_scale=1.0,
        optim="Adam",
        update_num=2,
        update_start=32,
        update_freq=1,
        NpS=20,
        action_noise="pn",
        update_noise="gs",
        buffer_noise="sp",
        buffer_size=None,
        model_type="hyper",
        llm_name="gpt2",
        fine_tune=False,
        out_bias=True,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = HyperLLMSolution(
            noise_dim,
            self.n_a,
            self.d,
            prior_scale=prior_scale,
            posterior_scale=posterior_scale,
            lr=lr,
            NpS=NpS,
            batch_size=batch_size,
            optim=optim,
            noise_coef=z_coef,
            based_weight_decay=based_weight_decay,
            hyper_weight_decay=hyper_weight_decay,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            buffer_size=buffer_size,
            model_type=model_type,
            llm_name=llm_name,
            fine_tune=fine_tune,
            out_bias=out_bias,
        )

        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
        history_action = np.zeros((T, self.env.n_actions), dtype=np.int32)
        true_action = np.zeros((T, self.env.n_actions), dtype=np.int32)
        for t in tqdm(range(T)):
            self.set_context()
            input_ids, attention_mask = self.env.get_feature()
            a_t = model.predict(input_ids, attention_mask, num=self.n_a)
            r_t = self.reward(a_t)
            regret_t = self.expect_regret(a_t, self.features)
            reward[t], expected_regret[t] = r_t.mean(), regret_t.mean()
            history_action[t] = a_t
            true_action[t] = self.env.true_action

            transitions = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "a": a_t,
                "r": r_t,
            }
            model.put(transitions)
            # update hypermodel
            update_results = {}
            if t >= update_start and (t + 1) % update_freq == 0:
                for _ in range(update_num):
                    update_results = model.update()
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.record("accuracy", np.sum(history_action[:t+1] == true_action[:t+1]) / ((t + 1) * self.env.n_actions))
                logger.record("action", np.sum(history_action) / ((t + 1) * self.env.n_actions))
                logger.record("reward", reward[t])
                logger.record("regret", expected_regret[t])
                for key, value in update_results.items():
                    logger.record(key, value)
                logger.dump(t)
        return reward, expected_regret

    def Hyper(
        self,
        T,
        logger,
        noise_dim=2,
        lr=0.01,
        based_weight_decay=0.0,
        hyper_weight_decay=0.0,
        z_coef=None,
        batch_size=32,
        hidden_sizes=(),
        prior_scale=1.0,
        optim="Adam",
        update_num=2,
        update_start=32,
        update_freq=1,
        NpS=20,
        action_noise="pn",
        update_noise="gs",
        buffer_noise="sp",
        buffer_size=None,
        out_bias=True,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = HyperSolution(
            noise_dim,
            self.n_a,
            self.d,
            hidden_sizes=hidden_sizes,
            prior_scale=prior_scale,
            lr=lr,
            batch_size=batch_size,
            optim=optim,
            noise_coef=z_coef,
            based_weight_decay=based_weight_decay,
            hyper_weight_decay=hyper_weight_decay,
            buffer_size=buffer_size,
            NpS=NpS,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            model_type="hyper",
            out_bias=out_bias,
        )

        log_interval = T // 1000
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
        for t in range(T):
            self.set_context()
            value = model.predict(self.features)
            a_t = rd_argmax(value)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"s": self.features, "r": r_t, "a": a_t}
            model.put(transitions)
            # update hypermodel
            if t >= update_start and (t + 1) % update_freq == 0:
                for _ in range(update_num):
                    model.update()
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.dump(t)
        return reward, expected_regret

    def EpiNet(
        self,
        T,
        logger,
        noise_dim=2,
        lr=0.01,
        based_weight_decay=0.0,
        hyper_weight_decay=0.0,
        z_coef=None,
        batch_size=32,
        hidden_sizes=(),
        prior_scale=1.0,
        optim="Adam",
        update_num=2,
        update_start=32,
        update_freq=1,
        NpS=20,
        action_noise="gs",
        update_noise="gs",
        buffer_noise="sp",
        buffer_size=None,
        class_num=1,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = HyperSolution(
            noise_dim,
            self.n_a,
            self.d,
            hidden_sizes=hidden_sizes,
            class_num=class_num,
            prior_scale=prior_scale,
            lr=lr,
            batch_size=batch_size,
            optim=optim,
            noise_coef=z_coef,
            based_weight_decay=based_weight_decay,
            hyper_weight_decay=hyper_weight_decay,
            buffer_size=buffer_size,
            NpS=NpS,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            model_type="epinet",
        )

        log_interval = T // 1000
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
        for t in range(T):
            self.set_context()
            value = model.predict(self.features)
            if class_num > 1:
                value = value[:, 1]
            a_t = rd_argmax(value)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"s": self.features, "r": r_t, "a": a_t}
            model.put(transitions)
            # update hypermodel
            if t >= update_start and (t + 1) % update_freq == 0:
                for _ in range(update_num):
                    model.update()
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.dump(t)
        return reward, expected_regret

    def Ensemble(
        self,
        T,
        logger,
        noise_dim=2,
        lr=0.01,
        based_weight_decay=0.0,
        hyper_weight_decay=0.0,
        z_coef=None,
        batch_size=32,
        hidden_sizes=(),
        prior_scale=1.0,
        optim="Adam",
        update_num=2,
        update_start=32,
        update_freq=1,
        NpS=20,
        action_noise="oh",
        update_noise="oh",
        buffer_noise="gs",
        buffer_size=None,
        out_bias=True,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = HyperSolution(
            noise_dim,
            self.n_a,
            self.d,
            hidden_sizes=hidden_sizes,
            prior_scale=prior_scale,
            lr=lr,
            batch_size=batch_size,
            optim=optim,
            noise_coef=z_coef,
            based_weight_decay=based_weight_decay,
            hyper_weight_decay=hyper_weight_decay,
            buffer_size=buffer_size,
            NpS=NpS,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            model_type="ensemble",
            out_bias=out_bias,
        )

        log_interval = T // 1000
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
        for t in range(T):
            self.set_context()
            value = model.predict(self.features)
            a_t = rd_argmax(value)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"s": self.features, "r": r_t, "a": a_t}
            model.put(transitions)
            # update hypermodel
            if t >= update_start and (t + 1) % update_freq == 0:
                for _ in range(update_num):
                    model.update()
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.dump(t)
        return reward, expected_regret

    def LMCTS(
        self,
        T,
        logger,
        noise_dim=2,
        lr=0.01,
        based_weight_decay=0.0,
        hyper_weight_decay=0.0,
        z_coef=None,
        batch_size=32,
        hidden_sizes=(),
        prior_scale=1.0,
        optim="Adam",
        update_num=2,
        update_start=32,
        update_freq=1,
        NpS=20,
        action_noise="oh",
        update_noise="oh",
        buffer_noise="gs",
        buffer_size=None,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = LMCTS(
            noise_dim,
            self.n_a,
            self.d,
            hidden_sizes=hidden_sizes,
            prior_scale=prior_scale,
            lr=lr,
            batch_size=batch_size,
            optim=optim,
            noise_coef=z_coef,
            based_weight_decay=based_weight_decay,
            hyper_weight_decay=hyper_weight_decay,
            buffer_size=buffer_size,
            NpS=NpS,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            model_type="linear",
        )

        update_step = 0
        log_interval = T // 1000
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
        for t in range(T):
            self.set_context()
            value = model.predict(self.features)
            a_t = rd_argmax(value)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"s": self.features, "r": r_t, "a": a_t}
            model.put(transitions)
            # update hypermodel
            if t >= update_start and (t + 1) % update_freq == 0:
                if update_num == 0:
                    num_iter = min(t + 1, 100)
                else:
                    num_iter = update_num
                if update_num == 0 and update_step > 0 and update_step % 20 == 0:
                    model.optimizer.lr = model.lr / update_step
                for _ in range(num_iter):
                    model.update()
                update_step += 1
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.dump(t)
        return reward, expected_regret
