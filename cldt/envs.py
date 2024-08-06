import gymnasium as gym

from cldt.wrappers import StepAPICompatibility


def make_panda_env(env_name, render_mode=None):
    import panda_gym

    env = gym.make(env_name)
    if render_mode is not None:
        env = gym.make(env_name, render_mode=render_mode)
    return env


def setup_env(env_name, wrappers=[], render=False) -> gym.Env:
    render_mode = "human" if render else None

    if env_name == "hopper":
        env = gym.make("Hopper-v4", render_mode=render_mode)
    elif env_name == "halfcheetah":
        env = gym.make("HalfCheetah-v4", render_mode=render_mode)
    elif env_name == "fetch-reach-dense":
        # import gymnasium_robotics
        env = gym.make("FetchReachDense-v3", render_mode=render_mode)
    elif env_name == "panda-reach-dense":
        env = make_panda_env("PandaReachDense-v3", render_mode)
    elif env_name == "panda-reach-sparse":
        env = make_panda_env("PandaReach-v3", render_mode)
    elif env_name == "panda-push-dense":
        env = make_panda_env("PandaPushDense-v3", render_mode)
    elif env_name == "panda-push-sparse":
        env = make_panda_env("PandaPush-v3", render_mode)
    elif env_name == "panda-pick-and-place-dense":
        env = make_panda_env("PandaPickAndPlaceDense-v3", render_mode)
    elif env_name == "panda-pick-and-place-sparse":
        env = make_panda_env("PandaPickAndPlace-v3", render_mode)
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")

    if wrappers is not None and len(wrappers) > 0:
        env = wrap_env(env, wrappers)

    return env


def wrap_env(env, wrappers):
    for wrapper_type in reversed(wrappers):
        if wrapper_type == "time-feature":
            from sb3_contrib.common.wrappers import TimeFeatureWrapper

            w = TimeFeatureWrapper
        elif wrapper_type == "step-api-compatibility":
            w = StepAPICompatibility
        else:
            raise ValueError(f"Unsupported wrapper type: {wrapper_type}")

        print(f"Wrapping environment with {wrapper_type}...")
        env = w(env)

    return env
