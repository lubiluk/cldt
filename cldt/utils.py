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
    for k in ["policy_kwargs", "training_kwargs", "eval_kwargs"]:
        if k in config and config[k] is not None:
            config[k] = eval(config[k])

    # Load the config
    if args.config is not None:
        base_config = load_config(args.config)
        config = extend_config(base_config, config)

    del config["config"]

    return config


def split_dataset(dataset_path, expert_size_ratio=0.3, number_of_samples=None):
    expert_dataset_path = dataset_path.replace(".pkl", "_expert.pkl")
    random_dataset_path = dataset_path.replace(".pkl", "_random.pkl")
    with open(expert_dataset_path, "rb") as f:
        expert_dataset = pickle.load(f)
    with open(random_dataset_path, "rb") as f:
        random_dataset = pickle.load(f)

    if len(expert_dataset) != len(random_dataset) and expert_size_ratio not in [0, 1]:
        number_of_samples = len(expert_dataset) + len(random_dataset)
        # smaller_dataset = min(len(expert_dataset), len(random_dataset))
        expert_size_ratio = len(expert_dataset) / number_of_samples

    if expert_size_ratio == 1:
        number_of_samples = len(expert_dataset)
        expert_size = number_of_samples
        random_size = 0
    elif expert_size_ratio == 0:
        number_of_samples = len(random_dataset)
        expert_size = 0
        random_size = number_of_samples
    else:
        number_of_samples = number_of_samples if number_of_samples is not None else len(expert_dataset)
        expert_size = int(number_of_samples * expert_size_ratio)
        random_size = int(number_of_samples * (1 - expert_size_ratio))

    random_dataset = sample(random_dataset, random_size)
    expert_dataset = sample(expert_dataset, expert_size)
    final_dataset = expert_dataset + random_dataset

    random.shuffle(final_dataset)

    final_dataset_path = dataset_path.replace(".pkl",
                                              f"{number_of_samples / 1000}k_expert_ratio_{expert_size_ratio}.pkl")

    with open(final_dataset_path, "wb") as f:
        pickle.dump(final_dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=False,
        default='datasets/panda_push_sparse.pkl',
        help="path to the dataset",
    )

    parser.add_argument(
        "-ratio",
        "--expert_size_ratio",
        type=float,
        required=False,
        default=0.1,
        help="ratio between expert and random samples",
    )

    parser.add_argument(
        "-s",
        "--number_of_samples",
        type=float,
        required=False,
        default=None,
        help="number of samples which be used in training",
    )

    args = parser.parse_args()

    split_dataset(args.dataset, args.expert_size_ratio, args.number_of_samples)
