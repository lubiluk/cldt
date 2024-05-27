import gymnasium as gym


def make_env(env_name, seed=None, render=False) -> gym.Env:
    if env_name == "hopper":
        env_name = "Hopper-v4"

    env = gym.make(env_name)
    if hasattr(env, "seed") and seed is not None:
        env.seed(seed)
    return env
