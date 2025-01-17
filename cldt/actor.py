"""
Functions for loading know actor
"""

from abc import ABC, abstractmethod


def type_2_class(actor_type):
    if actor_type == "random":
        from cldt.actors.random_actor import RandomActor

        return RandomActor
    elif actor_type == "dt":
        from cldt.actors.decision_transformer_actor import DecisionTransformerActor

        return DecisionTransformerActor
    elif actor_type == "reach":
        from cldt.actors.reach_actor import ReachActor

        return ReachActor
    elif actor_type == "tqc+her":
        from cldt.actors.tqc_her_actor import TqcHerActor

        return TqcHerActor
    elif actor_type == "nanodt":
        from cldt.actors.nano_dt_actor import NanoDTActor

        return NanoDTActor
    else:
        raise ValueError(f"Unknown policy type: {actor_type}")


def load_actor(actor_type, load_path, env=None):
    actor_class = type_2_class(actor_type)

    if env is not None:
        # Some actor from SB3 need env...
        return actor_class.load(load_path, env)

    return actor_class.load(load_path)


def setup_actor(actor_type, **kwargs):
    actor_class = type_2_class(actor_type)
    return actor_class(**kwargs)


class Actor(ABC):
    def __init__(self, device="auto") -> None:
        super().__init__()
        self.device = device

    def learn_offline(self, dataset, observation_space, action_space):
        raise NotImplementedError

    def learn_online(self, env):
        raise NotImplementedError

    def reset(self):
        pass

    def act(self, obs):
        raise NotImplementedError

    def evaluate(self, env, num_timesteps=1000, max_ep_len=None, render=False):
        """dont use this function, use evaluate_actor instead"""

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
    def load(path, env=None, device="auto"):
        raise NotImplementedError

    def save(self, path):
        pass


def evaluate_actor(
    actor,
    env,
    num_timesteps=1000,
    max_ep_len=None,
    render=False,
    extractor=None,
    **kwargs,
):
    returns = []
    ep_lens = []
    successes = []

    done = True

    for t in range(num_timesteps):
        if done:
            actor.reset(**kwargs)
            obs, _ = env.reset()
            done = False
            ep_ret = 0
            ep_len = 0
            done = False

        if render:
            env.render()

        if extractor is not None:
            obs = extractor(obs)

        act = actor.act(obs)

        res = env.step(act)

        if len(res) == 4:
            obs, rew, done, info = res
        else:
            obs, rew, ter, tru, info = res
            done = ter or tru

        ep_ret += rew
        ep_len += 1

        if max_ep_len and ep_len >= max_ep_len:
            done = True

        if done:
            returns.append(ep_ret)
            ep_lens.append(ep_len)
            successes.append(info.get("is_success", False))

    return returns, ep_lens, successes
