# Assumptions:
# 1. You can provide any Gym env
# 2. Optionally you can use cached experience (we will probably want a unified experience cache files)
# 3. You can use Decision Transformer or Bechavioral cloning
# 4. In case of DT, the training data will be automatically prepared with return-to-go
# 5. Probably we need to specify observation extractor
# 6. Function defined here should be usable in hyper-parameter search

import argparse
import pickle

import torch

from cldt.datasets import extract_dataset, load_dataset
from cldt.envs import setup_env
from cldt.actor import setup_actor, evaluate_actor
from cldt.utils import config_from_args, seed_env, seed_libraries
from paths import DATA_PATH


def train_single(
    actor_type,
    seed,
    env=None,
    wrappers=None,
    dataset=None,
    extractor_type=None,
    save_path=None,
    render=False,
    model_kwargs=None,
    training_kwargs=None,
    eval_kwargs=None,
    log_dir=None,
    device="auto"
):
    if model_kwargs is None:
        model_kwargs = {}
    if training_kwargs is None:
        training_kwargs = {}
    if eval_kwargs is None:
        eval_kwargs = {}

    # Save for printing
    env_name = env
    dataset_path = dataset

    # Set the seed
    seed_libraries(seed)
    # Create the environment
    env = setup_env(env, wrappers=wrappers, render=render)
    # Seed the environment
    seed_env(env, seed)

    print("Actor type:", actor_type)

    # Setup the policy that we will train
    actor = setup_actor(actor_type=actor_type, **model_kwargs)

    if dataset is not None:
        dataset = load_dataset(dataset)
        observation_space = env.observation_space

        if extractor_type is not None:
            dataset, observation_space = extract_dataset(
                dataset=dataset,
                extractor_type=extractor_type,
                observation_space=env.observation_space,
            )

        # Train the policy
        print(f"Training offline using dataset {dataset_path}...")
        actor.learn_offline(
            dataset=dataset,
            observation_space=observation_space,
            action_space=env.action_space,
            **training_kwargs,
        )
    else:
        # Train the policy
        print(f"Training online on {env_name}...")
        actor.learn_online(env=env, **training_kwargs)

    print("Training done!")

    # Save the policy
    if save_path is not None:
        save_path = f"{DATA_PATH}/{save_path}_seed_{seed}"
        actor.save(path=save_path)
        print(f"Policy saved to {save_path}")

    # Evaluate the policy
    print("Evaluating the policy...")
    score = evaluate_actor(actor=actor, env=env, render=render, **eval_kwargs)
    print(f"Score: {score}")

    env.close()

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        required=False,
        help="name of the environment",
    )
    parser.add_argument(
        "-w",
        "--wrappers",
        action="append",
        default=None,
        help="additional env wrappers",
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
        "-x",
        "--extractor-type",
        type=str,
        required=False,
        default=None,
        help="type of the extractor",
    )
    parser.add_argument(
        "-t",
        "--actor-type",
        type=str,
        required=False,
        help="actor type (list of available policies in cldt/actors.py)",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        required=False,
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
        help="kwargs for the policy",
    )
    parser.add_argument(
        "--training-kwargs",
        type=str,
        required=False,
        help="kwargs for the training",
    )
    parser.add_argument(
        "--eval-kwargs",
        type=str,
        required=False,
        help="kwargs for the evaluation",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default="configs/dt_panda_pick_and_place_dense_tf.yaml",
        help="path to the config file",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=False,
        help="directory to save logs",
    )

    args = parser.parse_args()

    config = config_from_args(args)

    print("Config:")
    print(config)

    print(torch.__version__)

    train_single(**config)
