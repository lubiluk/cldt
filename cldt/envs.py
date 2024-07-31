import gymnasium as gym

from cldt.wrappers import StepAPICompatibility


def setup_env(env_name, render=False) -> gym.Env:
    render_mode = "human" if render else None

    if env_name == "hopper":
        env_name = "Hopper-v4"
        return StepAPICompatibility(gym.make(env_name, render_mode=render_mode))
    elif env_name == "halfcheetah":
        env_name = "HalfCheetah-v4"
        return StepAPICompatibility(gym.make(env_name, render_mode=render_mode))
    elif env_name == "fetch-reach-dense":
        import gymnasium_robotics

        env_name = "FetchReachDense-v3"
        return StepAPICompatibility(gym.make(env_name, render_mode=render_mode))
    elif env_name == "panda-reach-dense":
        import panda_gym

        env_name = "PandaReachDense-v3"
        if render_mode is not None:
            return StepAPICompatibility(gym.make(env_name, render_mode=render_mode))

        return StepAPICompatibility(gym.make(env_name))
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")

    return env
