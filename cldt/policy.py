"""
Functions for loading know policies
"""

from abc import ABC, abstractmethod


def type_2_class(policy_type):
    if policy_type == "random":
        from cldt.policies.random_policy import RandomPolicy

        return RandomPolicy
    elif policy_type == "dt":
        from cldt.policies.decision_transformer_policy import DecisionTransformerPolicy

        return DecisionTransformerPolicy
    elif policy_type == "reach":
        from cldt.policies.reach_policy import ReachPolicy

        return ReachPolicy
    elif policy_type == "tqc+her":
        from cldt.policies.tqc_her_policy import TqcHerPolicy

        return TqcHerPolicy
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def load_policy(policy_type, load_path, env=None):
    policy_class = type_2_class(policy_type)
    
    if env is not None:
        # Some policies from SB3 need env...
        return policy_class.load(load_path, env)
    
    return policy_class.load(load_path)


def setup_policy(policy_type, **kwargs):
    policy_class = type_2_class(policy_type)
    return policy_class(**kwargs)


class Policy(ABC):
    def __init__(self, extractor=None) -> None:
        super().__init__()
        self.extractor = extractor

    def learn_offline(self, dataset, observation_space, action_space):
        raise NotImplementedError

    def learn_online(self, env):
        raise NotImplementedError

    def reset(self):
        pass

    def act(self, obs):
        raise NotImplementedError

    def evaluate(self, env, num_episodes=1, max_ep_len=None, render=False):
        returns = []
        ep_lens = []

        for _ in range(num_episodes):
            obs, _ = env.reset()

            done = False
            ep_len = 0
            ep_ret = 0

            while not done:
                if render:
                    env.render()

                act = self.act(obs)

                res = env.step(act)

                if len(res) == 4:
                    obs2, rew, done, _ = res
                else:
                    obs2, rew, ter, tru, _ = res
                    done = ter or tru

                ep_ret += rew
                ep_len += 1
                obs = obs2
                if max_ep_len and ep_len >= max_ep_len:
                    break

            returns.append(ep_ret)
            ep_lens.append(ep_len)

        return returns, ep_lens

    @staticmethod
    def load(path, env=None):
        raise NotImplementedError

    def save(self, path):
        pass
