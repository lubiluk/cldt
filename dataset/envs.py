import gymnasium as gym


def make_env(env_name, render=False) -> gym.Env:
    if env_name == "hopper":
        env_name = "Hopper-v4"

    render_mode = "human" if render else None

    env = gym.make(env_name, render_mode=render_mode)

    return env
