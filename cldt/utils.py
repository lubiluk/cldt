import argparse
import pickle
import random
import numpy as np
import torch
import yaml
from random import sample


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


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def extend_config(base_config, extend_config):
    for key, value in extend_config.items():
        if value is not None:
            base_config[key] = value
    return base_config


def config_from_args(args):
    config = vars(args)

    # Deserialize the kwargs
    for k in ["model_kwargs", "training_kwargs", "eval_kwargs"]:
        if k in config and config[k] is not None:
            config[k] = eval(config[k])

    # Load the config
    if args.config is not None:
        base_config = load_config(args.config)
        config = extend_config(base_config, config)

    del config["config"]

    return config

def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"

