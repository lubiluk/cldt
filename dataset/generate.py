"""
A program that generates a dataset by running a given trained model
on a given environment, recording observations, actions, rewards, terminals, and truncations.
The dataset is saved in a .pkl file of a dictionary with keys 'observations',
'next_observations', 'actions', 'rewards', 'terminals', and 'truncations'.
The program can be run from the command line with the following arguments:
-t/--policy-type: policy type (list of available policies in policies/__init__.py)
-p/--policy-path: path to the trained model
-e/--env: name of the environment
-n/--num-episodes: number of episodes to run
-o/--output-path: path to save the dataset
--render: whether to render the environment while generating the dataset
--seed: seed for the environment
Example usage:
python generate.py -t random -p cache/ppo -e hopper -n 1000 -o cache/hopper.pkl --render --seed 0
"""

import argparse
import importlib
import gymnasium as gym
import torch
import pickle
import numpy as np

from dataset.envs import make_env
from dataset.policies import load_policy


def generate_dataset(
    policy_type, policy_path, env_name, num_episodes, output_path, render, seed
):
    # Set the seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create the environment
    env = make_env(env_name, render)

    if hasattr(env.unwrapped, "seed") and seed is not None:
        env.unwrapped.seed(seed)

    # Load the policy
    policy = load_policy(policy_type, policy_path, env)

    # Initialize the dataset
    dataset = {
        "observations": [],
        "next_observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "truncations": [],
    }

    # Generate the dataset
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action, _ = policy.act(obs, deterministic=True)
            next_obs, reward, ter, tru, _ = env.step(action)
            dataset["observations"].append(obs)
            dataset["next_observations"].append(next_obs)
            dataset["actions"].append(action)
            dataset["rewards"].append(reward)
            dataset["terminals"].append(ter)
            dataset["truncations"].append(tru)
            obs = next_obs
            done = ter or tru

    # Save the dataset
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    # Close the environment
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--policy-type",
        type=str,
        required=True,
        help="policy type (list of available policies in policies/__init__.py)",
    )
    parser.add_argument(
        "-p",
        "--policy-path",
        type=str,
        default=None,
        help="path to the trained model",
    )
    parser.add_argument(
        "-e", "--env", type=str, required=True, help="name of the environment"
    )
    parser.add_argument(
        "-n",
        "--num-episodes",
        type=int,
        required=True,
        help="number of episodes to run",
    )
    parser.add_argument(
        "-o", "--output-path", type=str, required=True, help="path to save the dataset"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="whether to render the environment while generating the dataset",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="seed for the environment"
    )
    args = parser.parse_args()
    generate_dataset(
        args.policy_type,
        args.policy_path,
        args.env,
        args.num_episodes,
        args.output_path,
        args.render,
        args.seed,
    )
