from dataclasses import dataclass
from pathlib import Path
import pickle
import random

import numpy as np
import torch
import yaml
from cldt.extractors import DictExtractor
from cldt.policy import Policy

from transformers import (
    DecisionTransformerConfig,
    DecisionTransformerModel,
    Trainer,
    TrainingArguments,
)

from paths import DATA_PATH


@dataclass
class DecisionTransformerGymDataCollator:
    return_tensors: str = "pt"
    max_len: int = 20  # subsets of the episode we use for training
    state_dim: int = 17  # size of state space
    act_dim: int = 6  # size of action space
    max_ep_len: int = 1000  # max episode length in the dataset
    scale: float = 1000.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0  # to store the number of trajectories in the dataset

    def __init__(self, dataset, max_len, max_ep_len, scale, extractor=None) -> None:
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = (
            extractor.n_features
            if extractor is not None
            else len(dataset[0]["observations"][0])
        )
        self.dataset = dataset
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.scale = scale
        self.extractor = extractor

        # calculate dataset stats for normalization of states
        states = []
        traj_lens = []
        for path in dataset:
            obs = (
                self.extractor(path["observations"])
                if self.extractor is not None
                else path["observations"]
            )
            states.extend(obs)
            traj_lens.append(len(obs))
        self.n_traj = len(traj_lens)
        states = np.vstack(states)
        self.state_mean, self.state_std = (
            np.mean(states, axis=0),
            np.std(states, axis=0) + 1e-6,
        )

        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

        for ind in batch_inds:
            # for feature in features:
            feature = self.dataset[int(ind)]
            si = random.randint(0, len(feature["rewards"]) - 1)

            # extract observations
            obs = (
                self.extractor(feature["observations"])
                if self.extractor is not None
                else feature["observations"]
            )

            # get sequences from dataset
            s.append(
                np.array(obs[si : si + self.max_len]).reshape(1, -1, self.state_dim)
            )
            a.append(
                np.array(feature["actions"][si : si + self.max_len]).reshape(
                    1, -1, self.act_dim
                )
            )
            r.append(
                np.array(feature["rewards"][si : si + self.max_len]).reshape(1, -1, 1)
            )

            # d.append(
            #     np.array(feature["dones"][si : si + self.max_len]).reshape(1, -1)
            # )
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = (
                self.max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=1.0)[
                    : s[-1].shape[1]  # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1
            )
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1
            )
            # d[-1] = np.concatenate(
            #     [np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1
            # )
            rtg[-1] = (
                np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1)
                / self.scale
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        # d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

        loss = torch.mean((action_preds - action_targets) ** 2)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)


class DecisionTransformerPolicy(Policy):
    model = None
    return_scale = 1000.0
    K = 20
    max_ep_len = 1000
    scale = 1000
    hidden_size = 128
    n_layer = 3
    n_head = 1
    activation_function = "relu"
    dropout = 0.1
    extractor = None

    def __init__(
        self,
        return_scale,
        K,
        max_ep_len,
        scale,
        hidden_size,
        n_layer,
        n_head,
        activation_function,
        dropout,
        extractor_type=None,
        device="cuda",
    ) -> None:
        super().__init__()
        self.return_scale = return_scale
        self.K = K
        self.max_ep_len = max_ep_len
        self.scale = scale
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.activation_function = activation_function
        self.dropout = dropout
        self.extractor_type = extractor_type

    def learn_offline(
        self,
        dataset,
        observation_space,
        action_space,
        num_epochs=20,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        optimizer="adamw_torch",
        max_grad_norm=0.25,
    ):
        # TODO: support training a pretrained model
        if self.model is None:
            if self.extractor_type == "dict":
                self.extractor = DictExtractor(observation_space)
            else:
                self.extractor = None

            collator = DecisionTransformerGymDataCollator(
                dataset=dataset,
                max_len=self.K,
                max_ep_len=self.max_ep_len,
                scale=self.scale,
                extractor=self.extractor,
            )
            config = DecisionTransformerConfig(
                state_dim=collator.state_dim,
                act_dim=collator.act_dim,
                hidden_size=self.hidden_size,
                n_layer=self.n_layer,
                n_head=self.n_head,
                activation_function=self.activation_function,
                resid_pdrop=self.dropout,
                attn_pdrop=self.dropout
            )
            self.model = TrainableDT(config)

        training_args = TrainingArguments(
            output_dir=f"{DATA_PATH}/output/",
            remove_unused_columns=False,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            optim=optimizer,
            max_grad_norm=max_grad_norm,
            save_steps=5000
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
        )

        trainer.train()

        # Store mean, std
        self.state_mean = collator.state_mean.astype(np.float32)
        self.state_std = collator.state_std.astype(np.float32)

    # Function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
    def _get_action(self, states, actions, rewards, returns_to_go, timesteps):
        model = self.model
        # This implementation does not condition on past rewards

        states = states.reshape(1, -1, model.config.state_dim)
        actions = actions.reshape(1, -1, model.config.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:, -model.config.max_length :]
        actions = actions[:, -model.config.max_length :]
        returns_to_go = returns_to_go[:, -model.config.max_length :]
        timesteps = timesteps[:, -model.config.max_length :]
        padding = model.config.max_length - states.shape[1]
        # pad all tokens to sequence length
        attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
        states = torch.cat(
            [torch.zeros((1, padding, model.config.state_dim)), states], dim=1
        ).float()
        actions = torch.cat(
            [torch.zeros((1, padding, model.config.act_dim)), actions], dim=1
        ).float()
        returns_to_go = torch.cat(
            [torch.zeros((1, padding, 1)), returns_to_go], dim=1
        ).float()
        timesteps = torch.cat(
            [torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1
        )

        state_preds, action_preds, return_preds = model.original_forward(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return action_preds[0, -1]

    def reset(self, goal_return):
        device = self.model.device
        self.target_return = torch.tensor(
            goal_return, device=device, dtype=torch.float32
        ).reshape(1, 1)
        self.states = torch.zeros((1, self.model.config.state_dim)).to(
            device=device, dtype=torch.float32
        )
        self.actions = torch.zeros(
            (0, self.model.config.act_dim), device=device, dtype=torch.float32
        )
        self.rewards = torch.zeros(0, device=device, dtype=torch.float32)
        self.timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        self.goal_return = goal_return / self.return_scale

    def act(self, obs, prev_reward, prev_done):
        if self.extractor is not None:
            state = self.extractor(obs)
        else:
            state = torch.from_numpy(obs)

    def evaluate(
        self, env, goal_return, num_timesteps=1000, max_ep_len=1000, render=False
    ):
        # TODO: ultimately this function should be removed
        # instead it should work with generic evaluate function
        device = self.model.device
        state_mean = self.state_mean
        state_std = self.state_std

        state_dim = (
            env.observation_space.shape[0]
            if self.extractor is None
            else self.extractor.n_features
        )
        act_dim = env.action_space.shape[0]

        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)

        returns = []
        ep_lens = []

        scale = self.return_scale
        goal_return /= scale

        done = True

        for _ in range(num_timesteps):
            if done:
                episode_return, episode_length = 0, 0
                state, _ = env.reset()
                done = False

                if self.extractor is not None:
                    state = self.extractor(state)
                else:
                    state = torch.from_numpy(state)

                target_return = torch.tensor(
                    goal_return, device=device, dtype=torch.float32
                ).reshape(1, 1)
                states = state.reshape(1, state_dim).to(device=device, dtype=torch.float32)
                actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
                rewards = torch.zeros(0, device=device, dtype=torch.float32)

                if render:
                    env.render()

                timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

                t = 0

            actions = torch.cat(
                [actions, torch.zeros((1, act_dim), device=device)], dim=0
            )
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = self._get_action(
                (states - state_mean) / state_std,
                actions,
                rewards,
                target_return,
                timesteps,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, _ = env.step(action)

            if self.extractor is not None:
                state = self.extractor(state)
            else:
                state = torch.from_numpy(state)

            cur_state = state.to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            pred_return = target_return[0, -1] - (reward / scale)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1),
                ],
                dim=1,
            )

            episode_return += reward
            episode_length += 1

            t += 1

            if max_ep_len and episode_length >= max_ep_len:
                done = True

            if done:
                returns.append(episode_return)
                ep_lens.append(episode_length)

        return returns, ep_lens

    @staticmethod
    def load(path, env=None):
        path = Path(path)
        # Load the params from the path
        with open(path / "params.yaml", "r") as file:
            params = yaml.safe_load(file)
        policy = DecisionTransformerPolicy(**params)

        # Load the state_mean
        with open(path / "state_mean.npy", "rb") as f:
            policy.state_mean = np.load(f)

        # Load the state_std
        with open(path / "state_std.npy", "rb") as f:
            policy.state_std = np.load(f)

        # Load the model from the path
        policy.model = TrainableDT.from_pretrained(path)

        # Unpickle the extractor if there
        if (path / "extractor.pkl").exists():
            with open(path / "extractor.pkl", "rb") as file:
                policy.extractor = pickle.load(file)

        return policy

    def save(self, path):
        path = Path(path)
        # Save the model to the path
        self.model.save_pretrained(path)

        # Save the params to a yaml file
        params = {
            "return_scale": self.return_scale,
            "K": self.K,
            "max_ep_len": self.max_ep_len,
            "scale": self.scale,
            "hidden_size": self.hidden_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "activation_function": self.activation_function,
            "dropout": self.dropout,
            "extractor_type": self.extractor_type,
        }
        # Save params to a yaml file
        with open(path / "params.yaml", "w") as file:
            yaml.dump(params, file)

        # Save the state_mean
        with open(path / "state_mean.npy", "wb") as f:
            np.save(f, self.state_mean)

        # Save the state_std
        with open(path / "state_std.npy", "wb") as f:
            np.save(f, self.state_std)

        # Pickle the extractor
        if self.extractor is not None:
            with open(path / "extractor.pkl", "wb") as file:
                pickle.dump(self.extractor, file)
