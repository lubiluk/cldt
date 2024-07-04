"""
Functions for loading know policies
"""

from abc import ABC


def setup_policy(policy_type, policy_path, env):
    if policy_type == "random":
        return RandomPolicy(env)

    raise ValueError(f"Unknown policy type: {policy_type}")


class Policy(ABC):
    def __init__(self, env) -> None:
        super().__init__()
        self.env = env

    def act(self, obs, deterministic=False):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class RandomPolicy(Policy):
    def act(self, obs, deterministic=False):
        return self.env.action_space.sample(), None

    def load(self, path):
        pass
