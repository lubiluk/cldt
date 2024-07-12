"""
Functions for loading know policies
"""

from abc import ABC, abstractmethod


def setup_policy(policy_type, env, load_path=None, **kwargs):
    # env can be a list of environments
    if policy_type == "random":
        policy = RandomPolicy(env, **kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    if load_path is not None:
        policy.load(load_path)

    return policy




class Policy(ABC):
    def __init__(self, env) -> None:
        super().__init__()
        self.env = env

    def act(self, obs, deterministic=False):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def load(self, file):
        raise NotImplementedError

    def save(self, file):
        raise NotImplementedError

    def collect_experience(self, num_episodes=1):
        dataset = []
        for episode in range(num_episodes):
            trajectory = {
                "observations": [],
                "next_observations": [],
                "actions": [],
                "rewards": [],
                "terminals": [],
                "truncations": [],
            }
            obs, _ = self.env.reset()
            done = False
            while not done:
                action, _ = self.act(obs, deterministic=True)
                next_obs, reward, ter, tru, _ = self.env.step(action)
                trajectory["observations"].append(obs)
                trajectory["next_observations"].append(next_obs)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["terminals"].append(ter)
                trajectory["truncations"].append(tru)
                obs = next_obs
                done = ter or tru
            dataset.append(trajectory)
        return dataset



class RandomPolicy(Policy):
    def act(self, obs, deterministic=False):
        return self.env.action_space.sample(), None

