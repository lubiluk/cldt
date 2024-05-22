import pickle
import random

import gymnasium as gym
import numpy as np
import torch

env_name = "hopper"
dataset = "medium"
K = 20
batch_size = 64
num_eval_episodes = 100
pct_traj = 1.0
device = "cpu"
embed_dim = 128
n_layer = 3
n_head = 1
dropout = 0.1
activation_function = "relu"
warmup_steps = 10_000
learning_rate = 1e-4
weight_decay = 1e-4
max_iters = 10
num_steps_per_iter = 10_000


def discount_cumsum(x, gamma):
    """
    Calculates the discounted cumulative sum of an array.

    Parameters:
    x (numpy.ndarray): The input array.
    gamma (float): The discount factor.

    Returns:
    numpy.ndarray: The discounted cumulative sum of the input array.
    """
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def experiment():
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

    # concatenate all trajectories into single arrays to compute mean/std
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f"Starting new experiment: {env_name} {dataset}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print("=" * 50)

    # The code below is selecting as many trajectories as possible,
    # starting from the one with the highest return, until the
    # total number of timesteps across all selected trajectories
    # reaches num_timesteps

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        """
        Generate a batch of data for training a model.

        The function first randomly selects batch_size trajectory indices from a range of trajectory
        indices. Then, it iterates over each selected index and retrieves the corresponding trajectory
        from a list of trajectories. It extracts the relevant sequences of observations, actions,
        rewards, dones, and timesteps from the trajectory and performs padding and normalization
        operations on them.

        Finally, the function converts the numpy arrays to PyTorch tensors and returns the batch of tensors.

        Overall, this function is responsible for generating a batch of training data with the specified
        batch size and maximum sequence length, preparing the data for training a model.

        Args:
            batch_size (int): The number of samples in each batch. Default is 256.
            max_len (int): The maximum length of each sequence in the batch. Default is K.

        Returns:
            tuple: A tuple containing the following tensors:
                - s (torch.Tensor): The observations tensor of shape (batch_size, max_len, state_dim).
                - a (torch.Tensor): The actions tensor of shape (batch_size, max_len, act_dim).
                - r (torch.Tensor): The rewards tensor of shape (batch_size, max_len, 1).
                - d (torch.Tensor): The dones tensor of shape (batch_size, max_len).
                - rtg (torch.Tensor): The return-to-go tensor of shape (batch_size, max_len, 1).
                - timesteps (torch.Tensor): The timesteps tensor of shape (batch_size, max_len).
                - mask (torch.Tensor): The mask tensor of shape (batch_size, max_len).

        """
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
            a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                    : s[-1].shape[1] + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew / scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f"target_{target_rew}_return_mean": np.mean(returns),
                f"target_{target_rew}_return_std": np.std(returns),
                f"target_{target_rew}_length_mean": np.mean(lengths),
                f"target_{target_rew}_length_std": np.std(lengths),
            }

        return fn

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=embed_dim,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * embed_dim,
        activation_function=activation_function,
        n_positions=1024,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
    )

    model = model.to(device=device)

    warmup_steps = warmup_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )

    for iter in range(max_iters):
        outputs = trainer.train_iteration(
            num_steps=num_steps_per_iter, iter_num=iter + 1, print_logs=True
        )
