"""
Functions for loading know agents
"""

from abc import ABC, abstractmethod


def type_2_class(agent_type):
    if agent_type == "random":
        from cldt.agents.random_policy import RandomPolicy

        return RandomPolicy
    elif agent_type == "dt":
        from cldt.agents.decision_transformer_agent import DecisionTransformerAgent

        return DecisionTransformerAgent
    elif agent_type == "reach":
        from cldt.agents.reach_policy import ReachPolicy

        return ReachPolicy
    elif agent_type == "tqc+her":
        from cldt.agents.tqc_her_policy import TqcHerPolicy

        return TqcHerPolicy
    elif agent_type == "nanodt":
        from cldt.agents.nano_dt_agent import NanoDTAgent
        
        return NanoDTAgent
    else:
        raise ValueError(f"Unknown policy type: {agent_type}")


def load_agent(agent_type, load_path, env=None):
    agent_class = type_2_class(agent_type)

    if env is not None:
        # Some agents from SB3 need env...
        return agent_class.load(load_path, env)

    return agent_class.load(load_path)


def agent_policy(agent_type, **kwargs):
    agent_class = type_2_class(agent_type)
    return agent_class(**kwargs)


class Agent(ABC):
    def __init__(self) -> None:
        super().__init__()

    def learn_offline(self, dataset, observation_space, action_space):
        raise NotImplementedError

    def learn_online(self, env):
        raise NotImplementedError

    def reset(self):
        pass

    def act(self, obs):
        raise NotImplementedError

    def evaluate(self, env, num_timesteps=1000, max_ep_len=None, render=False):
        """dont use this function, use evaluate_agent instead"""

        # TODO: This function is going to be moved outside of Policy class
        # It is here for now because DecisionTransformerPolicy needs to subclass it
        returns = []
        ep_lens = []
        successes = []

        done = True

        for t in range(num_timesteps):
            if done:
                obs, _ = env.reset()
                done = False
                ep_ret = 0
                ep_len = 0
                done = False

            if render:
                env.render()

            act = self.act(obs)

            res = env.step(act)

            if len(res) == 4:
                obs2, rew, done, info = res
            else:
                obs2, rew, ter, tru, info = res
                done = ter or tru

            ep_ret += rew
            ep_len += 1
            obs = obs2

            if max_ep_len and ep_len >= max_ep_len:
                done = True

            if done:
                returns.append(ep_ret)
                ep_lens.append(ep_len)
                successes.append(info.get("is_success", False))

        return returns, ep_lens, successes

    @staticmethod
    def load(path, env=None):
        raise NotImplementedError

    def save(self, path):
        pass


def evaluate_agent(agent, env, num_timesteps=1000, max_ep_len=None, render=False, **kwargs):
        returns = []
        ep_lens = []
        successes = []

        done = True

        for t in range(num_timesteps):
            if done:
                agent.reset(**kwargs)
                obs, _ = env.reset()
                done = False
                ep_ret = 0
                ep_len = 0
                done = False

            if render:
                env.render()

            act = self.act(obs)

            res = env.step(act)

            if len(res) == 4:
                obs2, rew, done, info = res
            else:
                obs2, rew, ter, tru, info = res
                done = ter or tru

            ep_ret += rew
            ep_len += 1
            obs = obs2

            if max_ep_len and ep_len >= max_ep_len:
                done = True

            if done:
                returns.append(ep_ret)
                ep_lens.append(ep_len)
                successes.append(info.get("is_success", False))

        return returns, ep_lens, successes
