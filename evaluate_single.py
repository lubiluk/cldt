import argparse
from statistics import mean

from cldt.envs import setup_env
from cldt.actor import load_actor, evaluate_actor
from cldt.extractors import setup_extractor
from cldt.utils import config_from_args, seed_env, seed_libraries
from paths import DATA_PATH


def evaluate_single(
    actor_type,
    save_path,
    seed,
    env,
    extractor_type=None,
    wrappers=None,
    render=False,
    eval_kwargs=None,
    **kwargs,
):
    if eval_kwargs is None:
        eval_kwargs = {}

    # Save for printing
    env_name = env
    load_path = f"{DATA_PATH}/{save_path}_seed_{seed}"
    # Set the seed
    seed_libraries(seed)
    # Create the environment
    env = setup_env(env, wrappers=wrappers, render=render)
    # Seed the environment
    seed_env(env, seed)

    if extractor_type is not None:
        extractor = setup_extractor(extractor_type, env.observation_space)
    else:
        extractor = None

    # Setup the actor that we will train
    actor = load_actor(actor_type=actor_type, load_path=load_path, env=env)

    # Evaluate the actor
    print(f"Evaluating the actor {actor_type} on {env_name}...")
    returns, eplens, successes = evaluate_actor(
        actor=actor, env=env, render=render, extractor=extractor, **eval_kwargs
    )
    mean_ret = mean(returns)
    mean_len = mean(eplens)
    success_rate = mean(successes)
    print(f"Mean return: {mean_ret}")
    print(f"Mean episode length: {mean_len}")
    print(f"Success rate: {success_rate}")

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
        "--actor-type",
        type=str,
        required=False,
        default=None,
        help="actor type (list of available actors in cldt/actors.py)",
    )
    parser.add_argument(
        "-l",
        "--save-path",
        type=str,
        required=False,
        default=None,
        help="path to load the actor from",
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
