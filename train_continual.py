from cldt.envs import setup_env
from cldt.actor import setup_actor
from cldt.utils import seed_env, seed_libraries


def train_continual(
    env_names,
    dataset_paths,
    policy_type,
    policy_save_path,
    seed,
    render,
    device,
):
    # Set the seed
    seed_libraries(seed)
    # Create the environments
    envs = [setup_env(env_name, render) for env_name in env_names]
    # Seed the environments
    for env in envs:
        seed_env(env, seed)
        
    policy = setup_actor(policy_type, env=envs, device=device)

    for env_name, dataset_path in zip(env_names, dataset_paths):
        # Train the policy on one task
        # train()
        # Evaluate the policy on the task
        # score = evaluate()
        # Report intermediate result

    # Evaluate again on all tasks
    for env_name in env_names:
        # evaluate()
