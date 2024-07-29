import argparse

from cldt.envs import setup_env
from cldt.policies import setup_policy
from cldt.utils import config_from_args, eval_config, seed_env, seed_libraries

# TODO: doesn't work yet
def evaluate_single(
    env, policy_type, load_path, seed, render=False, policy_kwargs={}, eval_kwargs={}
):
    # Save for printing
    env_name = env

    # Set the seed
    seed_libraries(seed)
    # Create the environment
    env = setup_env(env, render)
    # Seed the environment
    seed_env(env, seed)

    # Setup the policy that we will train
    policy = setup_policy(policy_type=policy_type, load_path=load_path, **policy_kwargs)

    # Evaluate the policy
    print(f"Evaluating the policy {policy_type} on {env_name}...")
    score = policy.evaluate(env=env, render=render, **eval_kwargs)
    print(f"Score: {score}")

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
        "-t",
        "--policy-type",
        type=str,
        required=False,
        default=None,
        help="policy type (list of available policies in cldt/policies.py)",
    )
    parser.add_argument(
        "-l",
        "--load-path",
        type=str,
        required=False,
        default=None,
        help="path to load the policy from",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=None, help="seed for the environment"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="whether to render the environment while evaluating",
    )
    parser.add_argument(
        "--eval-kwargs",
        type=str,
        required=False,
        default=None,
        help="kwargs for the evaluation",
    )
    args = parser.parse_args()

    config = config_from_args(args)

    print("Config:")
    # print(config)

    # evaluate_single(**config)
