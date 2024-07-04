# Assumptions:
# 1. You can provide any Gym env
# 2. Optionally you can use cached experience (we will probably want a unified experience cache files)
# 3. You can use Decision Transformer or Bechavioral cloning
# 4. In case of DT, the training data will be automatically prepared with return-to-go
# 5. Probably we need to specify observation extractor
# 6. Function defined here should be usable in hyper-parameter search

import argparse
import pickle
from cldt.envs import name2env, setup_env
from cldt.policies import setup_policy
from cldt.utils import seed_everything


def train(
    env_name: str,
    dataset_path,
    policy_type,
    policy_save_path,
    seed,
    render=False,
    device="cpu",
):
    # Set the seed
    seed_everything(seed)

    # Create the environment
    env = setup_env(env_name, render)

    if hasattr(env.unwrapped, "seed") and seed is not None:
        env.unwrapped.seed(seed)

    # Setup the policy that will generate episodes
    policy = setup_policy(policy_type=policy_type, env=env)

    # Load the dataset
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    print(f"Training policy {policy_type} on {env_name} using {dataset_path}...")
    policy.train(dataset=dataset, config=config)  # ???

    policy.save(path=policy_save_path)  # ???
    print(f"Policy saved to {policy_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        required=False,
        default="hopper",
        help="name of the environment",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=False,
        default=None,
        help="path to the dataset",
    )
    parser.add_argument(
        "-p",
        "--policy-type",
        type=str,
        required=True,
        help="policy type (list of available policies in cldt/policies.py)",
    )
    parser.add_argument(
        "-s",
        "--policy-save-path",
        type=str,
        required=True,
        help="path to save the policy",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="seed for the environment"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="whether to render the environment while evaluating",
    )
    parser.add_argument("--device", type=str, default="cpu", help="device to use")

    args = parser.parse_args()

    train(
        args.env,
        args.dataset,
        args.policy_type,
        args.policy_save_path,
        args.seed,
        args.render,
        args.device,
    )
