import pickle

import gymnasium as gym
import numpy as np

env_name = "hopper"
dataset = "medium"

# Hopper enviroment
env = gym.make("Hopper-v4")
max_ep_len = 1000
env_targets = [3600, 1800]  # evaluation conditioning targets
scale = 1000.0  # normalization for rewards/returns

state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

dataset_path = f"data/{env_name}-{dataset}-v2.pkl"
with open(dataset_path, "rb") as f:
    trajectories = pickle.load(f)


# save all path information into separate lists
mode = "normal"
states, traj_lens, returns = [], [], []
for path in trajectories:
    if mode == "delayed":  # delayed: all rewards moved to end of trajectory
        path["rewards"][-1] = path["rewards"].sum()
        path["rewards"][:-1] = 0.0
    states.append(path["observations"])
    traj_lens.append(len(path["observations"]))
    returns.append(path["rewards"].sum())
traj_lens, returns = np.array(traj_lens), np.array(returns)

# used for input normalization
states = np.concatenate(states, axis=0)
state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

num_timesteps = sum(traj_lens)

print("=" * 50)
print(f"Starting new experiment: {env_name} {dataset}")
print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
print("=" * 50)
