# cldt
Continual learning with decision transformer

## Downloading all datasets from D4RL

```bash
python -m dataset.download
```

## Generating a dataset

1. Add your policy to `dataset/policies.py`.
2. Add you environment to `dataset/envs.py`.
3. Run the generation module:

```bash
python -m dataset.generate -t [name of your poliyc] -p [path to the trained policy (if applicable)] -e [name of your env] -n 1000 -o [output file path] --render --seed 0

Example:
python -m dataset.generate -t random -p cache/random -e hopper -n 1000 -o cache/hopper.pkl --render --seed 0
```

## Downloading Atari datasets

```bash
gsutil -m cp -R gs://atari-replay-datasets/dqn/Breakout/ ./cache/
```

## Running Atari DT

```bash
python -m experiment.atari --seed 1234 --context-length 30 --epochs 5 --model-type reward_conditioned --num-steps 500000 --num-buffers 50 --game Breakout --batch-size 128
```