# Assumptions:
# 1. You can provide any Gym env
# 2. Optionally you can use cached experience (we will probably want a unified experience cache files)
# 3. You can use Decision Transformer or Bechavioral cloning
# 4. In case of DT, the training data will be automatically prepared with return-to-go
# 5. Probably we need to specify observation extractor
# 6. Function defined here should be usable in hyper-parameter search

import argparse
import pickle
from cldt.envs import setup_env
from cldt.policies import setup_policy
from cldt.utils import extend_config, load_config, seed_env, seed_libraries


def train_single(
    env,
    dataset,
    policy_type,
    save_path,
    seed,
    render=False,
    policy_kwargs={},
    training_kwargs={},
    eval_kwargs={},
):
    # Save for printing
    env_name = env
    dataset_path = dataset

    # Set the seed
    seed_libraries(seed)
    # Create the environment
    env = setup_env(env, render)
    # Seed the environment
    seed_env(env, seed)

    # Setup the policy that we will train
    policy = setup_policy(policy_type=policy_type, **policy_kwargs)

    # Load the dataset
    with open(dataset, "rb") as f:
        dataset = pickle.load(f)

    # Train the policy
    print(f"Training policy {policy_type} on {env_name} using {dataset_path}...")
    policy.learn(dataset=dataset, **training_kwargs)
    print("Training done!")

    # Evaluate the policy
    print("Evaluating the policy...")
    score = policy.evaluate(env=env, render=render, **eval_kwargs)
    print(f"Score: {score}")

    # Save the policy
    if save_path is not None:
        policy.save(path=save_path)
        print(f"Policy saved to {save_path}")

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        required=False,
        default=None,
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
        "-t",
        "--policy-type",
        type=str,
        required=False,
        default=None,
        help="policy type (list of available policies in cldt/policies.py)",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        required=False,
        default=None,
        help="path to save the policy",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="seed for the environment"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="whether to render the environment while evaluating",
    )
    parser.add_argument(
        "--policy-kwargs",
        type=str,
        required=False,
        default=None,
        help="kwargs for the policy",
    )
    parser.add_argument(
        "--training-kwargs",
        type=str,
        required=False,
        default=None,
        help="kwargs for the training",
    )
    parser.add_argument(
        "--eval-kwargs",
        type=str,
        required=False,
        default=None,
        help="kwargs for the evaluation",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default="configs/halfcheetah.yaml",
        help="path to the config file",
    )

    args = parser.parse_args()

    config = vars(args)

    # Deserialize the kwargs
    if config["policy_kwargs"] is not None:
        config["policy_kwargs"] = eval(config["policy_kwargs"])
    if config["training_kwargs"] is not None:
        config["training_kwargs"] = eval(config["training_kwargs"])
    if config["eval_kwargs"] is not None:
        config["eval_kwargs"] = eval(config["eval_kwargs"])

    # Load the config
    if args.config is not None:
        base_config = load_config(args.config)
        config = extend_config(base_config, config)

    del config["config"]

    print("Config:")
    print(config)

    train_single(**config)
