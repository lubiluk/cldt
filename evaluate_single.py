import argparse
from statistics import mean

from cldt.envs import setup_env
from cldt.agent import load_agent, evaluate_agent
from cldt.utils import config_from_args, seed_env, seed_libraries
from paths import DATA_PATH


def evaluate_single(
    agent_type,
    save_path,
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
    load_path = f'{DATA_PATH}/{save_path}_seed_{seed}'
    # Set the seed
    seed_libraries(seed)
    # Create the environment
    env = setup_env(env, wrappers=wrappers, render=render)
    # Seed the environment
    seed_env(env, seed)

    # Setup the agent that we will train
    agent = load_agent(agent_type=agent_type, load_path=load_path, env=env)

    # Evaluate the agent
    print(f"Evaluating the agent {agent_type} on {env_name}...")
    returns, eplens, successes = evaluate_agent(agent=agent, env=env, render=render, **eval_kwargs)
    mean_ret = mean(returns)
    mean_len = mean(eplens)
    success_rate = mean(successes)
    print(f"Mean return: {mean_ret}")
    print(f"Mean episode length: {mean_len}")
    print(f'Success rate: {success_rate}')

    env.close()

    return mean_ret


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
        "--agent-type",
        type=str,
        required=False,
        default=None,
        help="agent type (list of available agents in cldt/agents.py)",
    )
    parser.add_argument(
        "-l",
        "--save-path",
        type=str,
        required=False,
        default=None,
        help="path to load the agent from",
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
        help="path to the config file",
    )
    args = parser.parse_args()

    config = config_from_args(args)

    print("Config:")
    print(config)

    evaluate_single(**config)
