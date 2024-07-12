import gymnasium as gym


def setup_env(env_name, render=False) -> gym.Env:
    if env_name == "hopper":
        env_name = "Hopper-v4"
    elif env_name == "halfcheetah":
        env_name = "HalfCheetah-v4"
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")

    render_mode = "human" if render else None

    env = gym.make(env_name, render_mode=render_mode)

    return env
