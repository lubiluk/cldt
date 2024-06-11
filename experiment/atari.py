import argparse
import logging

from dataset.create_dataset import create_dataset
from policy.model_atari import GPT, GPTConfig
from policy.utils import set_seed
from torch.utils.data import Dataset
import torch
import numpy as np

from trainer.trainer_atari import Trainer, TrainerConfig


# TODO: Move somewhere else
class StateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(
            np.array(self.data[idx:done_idx]), dtype=torch.float32
        ).reshape(
            block_size, -1
        )  # (block_size, 4*84*84)
        states = states / 255.0
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(
            1
        )  # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(
            self.timesteps[idx : idx + 1], dtype=torch.int64
        ).unsqueeze(1)

        return states, actions, rtgs, timesteps


def main(
    seed,
    context_length,
    epochs,
    model_type,
    num_steps,
    num_buffers,
    game,
    batch_size,
    trajectories_per_buffer,
    data_dir_prefix,
):
    set_seed(seed)
    obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(
        args.num_buffers,
        args.num_steps,
        args.game,
        args.data_dir_prefix,
        args.trajectories_per_buffer,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    train_dataset = StateActionReturnDataset(
        obss, args.context_length * 3, actions, done_idxs, rtgs, timesteps
    )
    mconf = GPTConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        n_layer=6,
        n_head=8,
        n_embd=128,
        model_type=args.model_type,
        max_timestep=max(timesteps),
    )
    model = GPT(mconf)

    # initialize a trainer instance and kick off training
    epochs = args.epochs
    tconf = TrainerConfig(
        max_epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=2 * len(train_dataset) * args.context_length * 3,
        num_workers=4,
        seed=args.seed,
        model_type=args.model_type,
        game=args.game,
        max_timestep=max(timesteps),
    )
    trainer = Trainer(model, train_dataset, None, tconf)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--context-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model-type", type=str, default="reward_conditioned")
    parser.add_argument("--num-steps", type=int, default=500000)
    parser.add_argument("--num-buffers", type=int, default=50)
    parser.add_argument("--game", type=str, default="Breakout")
    parser.add_argument("--batch-size", type=int, default=128)
    #
    parser.add_argument(
        "--trajectories-per-buffer",
        type=int,
        default=10,
        help="Number of trajectories to sample from each of the buffers.",
    )
    parser.add_argument("--data-dir-prefix", type=str, default="./cache/")
    args = parser.parse_args()
    main(
        args.seed,
        args.context_length,
        args.epochs,
        args.model_type,
        args.num_steps,
        args.num_buffers,
        args.game,
        args.batch_size,
        args.trajectories_per_buffer,
        args.data_dir_prefix,
    )
