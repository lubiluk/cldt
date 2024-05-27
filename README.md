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
python generate.py -t [name of your poliyc] -p [path to the trained policy (if applicable)] -e [name of your env] -n 1000 -o [output file path] --render --seed 0

Example:
python generate.py -t random -p cache/random -e hopper -n 1000 -o cache/hopper.pkl --render --seed 0
```