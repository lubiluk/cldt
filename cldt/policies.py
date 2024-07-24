"""
Functions for loading know policies
"""

from abc import ABC, abstractmethod


def setup_policy(policy_type, load_path=None, **kwargs):
    # env can be a list of environments
    if policy_type == "random":
        from cldt.random_policy import RandomPolicy

        policy = RandomPolicy()
    elif policy_type == "dt":
        from cldt.decision_transformer import DecisionTransformer

        policy = DecisionTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    if load_path is not None:
        policy.load(load_path)

    return policy


class Policy(ABC):
    def __init__(self) -> None:
        super().__init__()

    def learn(self):
        raise NotImplementedError

    def evaluate(self, env, num_episodes=1, max_ep_len=None, record_trajectories=False):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError
