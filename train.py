# Assumptions:
# 1. You can provide any Gym env
# 2. Optionally you can use cached experience (we will probably want a unified experience cache files)
# 3. You can use Decision Transformer or Bechavioral cloning
# 4. In case of DT, the training data will be automatically prepared with return-to-go
# 5. Probably we need to specify observation extractor
# 6. Function defined here should be usable in hyper-parameter search

import fire

from cldt.envs import name2env


def train(env_name: str, cache_dir="./cache"):
    env = name2env(env_name)
    


if __name__ == "__main__":
    fire.Fire(train)
