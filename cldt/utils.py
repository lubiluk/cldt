import random
import numpy as np
import torch


def seed_libraries(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_env(env, seed):
    # env can be a list of environments or a single environment
    if env is not None:
        if isinstance(env, list):
            for e in env:
                if hasattr(e.unwrapped, "seed") and seed is not None:
                    e.unwrapped.seed(seed)
        else:
            if hasattr(env.unwrapped, "seed") and seed is not None:
                env.unwrapped.seed(seed)