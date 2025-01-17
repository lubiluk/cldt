from typing import Optional

import gymnasium.spaces as spaces
import numpy as np
import torch
from numpy.typing import NDArray


def setup_extractor(extractor_type, observation_space, **kwargs):
    if extractor_type == "dict":
        return DictExtractor(observation_space=observation_space, **kwargs)
    elif extractor_type is None:
        return None
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")


class DictExtractor:
    def __init__(self, observation_space: spaces.Dict, retained_keys=None) -> None:
        self.n_features = spaces.utils.flatdim(observation_space)
        self._keys = (
            retained_keys if retained_keys is not None else observation_space.keys()
        )
        self.observation_space = spaces.flatten_space(observation_space)

    def __call__(
        self, observation: dict[str, NDArray], device: Optional[torch.device] = None
    ):
        arrays = [
            observation[k]
            for k in self._keys
        ]

        return np.concatenate(arrays, axis=-1)
