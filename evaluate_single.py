import argparse
from statistics import mean

from cldt.envs import setup_env
from cldt.policy import load_policy
from cldt.utils import config_from_args, seed_env, seed_libraries
from paths import DATA_PATH


def evaluate_single(
    policy_type,
    load_path,
    seed,
    env,
    wrappers=None,
    render=False,
    eval_kwargs=None,
    **kwargs,
):
    if eval_kwargs is None:
        eval_kwargs = {}

    # Save for printing
    env_name = env
    load_path = f'{DATA_PATH}/{load_path}_seed_{seed}'
    # Set the seed
    seed_libraries(seed)
    # Create the environment
    env = setup_env(env, wrappers=wrappers, render=render)
    # Seed the environment
    seed_env(env, seed)

    # Setup the policy that we will train
    policy = load_policy(policy_type=policy_type, load_path=load_path, env=env)

    # Evaluate the policy
    print(f"Evaluating the policy {policy_type} on {env_name}...")
    returns, eplen = policy.evaluate(env=env, render=render, **eval_kwargs)
    score = mean(returns)
    lens = mean(eplen)
    print(f"Mean return: {score}")
    print(f"Mean episode length: {lens}")

    env.close()

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
        "-w",
        "--wrappers",
        nargs="+",
        type=list,
        default=None,
        help="additional env wrappers",
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
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default='configs/dt_panda_pick_and_place_dense_tf.yaml',
        help="path to the config file",
    )
    args = parser.parse_args()

    config = config_from_args(args)

    print("Config:")
    print(config)

    evaluate_single(**config)
