"""
Functions for loading know policies
"""

from abc import ABC, abstractmethod


def setup_policy(policy_type, load_path=None, extractor=None, **kwargs):
    # env can be a list of environments
    if policy_type == "random":
        from cldt.policy_types.random_policy import RandomPolicy

        policy = RandomPolicy()
    elif policy_type == "dt":
        from cldt.policy_types.decision_transformer import DecisionTransformer

        policy = DecisionTransformer(extractor=extractor, **kwargs)
    elif policy_type == "reach":
        from cldt.policy_types.reach_policy import ReachPolicy

        policy = ReachPolicy()
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    if load_path is not None:
        policy.load(load_path)

    return policy


class Policy(ABC):
    def __init__(self, extractor=None) -> None:
        super().__init__()
        self.extractor = extractor

    def learn(self, dataset=None, env=None):
        pass

    def evaluate(self, env, num_episodes=1, max_ep_len=None, render=False):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def save(self, path):
        pass
