from dataclasses import dataclass
from typing import Dict, List
import torch.nn as nn
from cldt.actor import Actor
import torch
from torch.utils.data import DataLoader
import numpy.typing as npt
import gymnasium.spaces as spaces


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, act_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def collate_fn(batch):
    return {
        # "id": torch.Tensor([x.id for x in batch]),
        # "seed": torch.Tensor([x.seed for x in batch]),
        # "total_steps": torch.Tensor([x.total_steps for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x["observations"], dtype=torch.float32) for x in batch],
            batch_first=True,
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x["actions"], dtype=torch.float32) for x in batch],
            batch_first=True,
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x["rewards"]) for x in batch], batch_first=True
        ),
        "terminals": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x["terminals"]) for x in batch], batch_first=True
        )
    }


@dataclass
class BehavioralCloningConfig:
    hidden_sizes: List[int]


@dataclass
class BehavioralCloningTrainerConfig:
    learning_rate: float = 1e-3
    num_epochs: int = 32
    batch_size: int = 256


class BCActor(Actor):
    def __init__(self, hidden_sizes, device="cpu"):
        self.device = device
        self.model = None
        self.model_config = BehavioralCloningConfig(hidden_sizes=hidden_sizes)

    def learn_offline(self, dataset, observation_space, action_space, **kwargs):
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        policy_net = PolicyNetwork(obs_dim, act_dim, self.model_config.hidden_sizes)

        self.trainer_config = BehavioralCloningTrainerConfig(**kwargs)

        dataloader = DataLoader(
            dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        optimizer = torch.optim.Adam(
            policy_net.parameters(), lr=self.trainer_config.learning_rate
        )
        loss_fn = nn.MSELoss()

        num_epochs = self.trainer_config.num_epochs

        for epoch in range(num_epochs):
            for batch in dataloader:
                a_pred = policy_net(batch["observations"])
                a_hat = batch["actions"]
                loss = loss_fn(a_pred, a_hat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")

        self.model = policy_net

    def act(self, observation: npt.ArrayLike):
        obs = torch.as_tensor(observation, dtype=torch.float32)

        with torch.no_grad():
            return self.model(obs).cpu().numpy()

    def save(self, path):
        if not path.endswith(".pth"):
            path += ".pth"

        torch.save(
            {
                "model_state_dict": self.model.to("cpu").state_dict(),
                "model_config": self.model_config,
                # "trainer_config": self.trainer_config,
            },
            path,
        )

    @classmethod
    def load(cls, path, env=None, device="cpu"):
        if not path.endswith(".pth"):
            path += ".pth"

        checkpoint = torch.load(path, map_location="cpu")
        # Extract the saved model configuration
        model_config = checkpoint["model_config"]

        actor = cls(model_config.hidden_sizes, device=device)
        input_dim =  spaces.utils.flatdim(env.observation_space)
        act_dim = env.action_space.shape[0]
        actor.model = PolicyNetwork(
            input_dim, act_dim, model_config.hidden_sizes
        )
        actor.model.load_state_dict(checkpoint["model_state_dict"])
        actor.model.to(device)
        actor.model.eval()

        return actor
