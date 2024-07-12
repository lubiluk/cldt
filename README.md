# cldt
Continual learning with decision transformer

## Downloading datasets from D4RL

Determine your desired dataset name, see dataset_infos.py. Then run the download script:

```bash
python download_dataset.py halfcheetah-expert-v2
```

## Training a Decision Transformer

Environment should be a supported one, see envs.py.
Policy should be a supported one, see policies.py

```bash
usage: generate_dataset.py [-h] [-t POLICY_TYPE] [-p POLICY_PATH] [-e ENV] [-n NUM_EPISODES] [-o OUTPUT_PATH] [--render] [--seed SEED]
example: python train_single.py -e halfcheetah -d cache/halfcheetah-expert-v2 -p dt -s trained/haflcheetah-dt --seed 1234
```

## Generating a dataset

1. Add your policy to `policies.py`.
2. Add you environment to `envs.py`.
3. Run the generation module:

```bash
python generate_dateset.py -t [name of your policy] -p [path to the trained policy (if applicable)] -e [name of your env] -n 1000 -o [output file path] --render --seed 0

Example:
python -m generate_dateset.py -t random -p cache/random -e hopper -n 1000 -o cache/hopper.pkl --render --seed 0
```


----

## Downloading Atari datasets

```bash
gsutil -m cp -R gs://atari-replay-datasets/dqn/Breakout/ ./cache/
```

## Running Atari DT

```bash
python -m experiment.atari --seed 1234 --context-length 30 --epochs 5 --model-type reward_conditioned --num-steps 500000 --num-buffers 50 --game Breakout --batch-size 128
```