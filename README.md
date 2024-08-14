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

You can either create a YAML config file or provide agruments directly. You can also provide a config file an overwrite it's values using arguments.

```bash
Usage:
usage: train_single.py [-h] [-e ENV] [-d DATASET] [-t POLICY_TYPE] [-s SAVE_PATH] [--seed SEED] [--render] [--policy-kwargs POLICY_KWARGS] [--training-kwargs TRAINING_KWARGS] [--eval-kwargs EVAL_KWARGS] [-c CONFIG]

Example:
python train_single.py -e halfcheetah -d cache/halfcheetah-expert-v2 -p dt -s trained/haflcheetah-dt --seed 1234
python train_single.py -c configs/halfcheetah.yaml
```

## Generating a dataset

1. Add your policy to `policies.py`.
2. Add you environment to `envs.py`.
3. Run the generation module:

```bash
Usage: 
generate_dataset.py [-h] [-t POLICY_TYPE] [-p POLICY_PATH] [-e ENV] [-n NUM_EPISODES] [-o OUTPUT_PATH] [--render] [--seed SEED]

Example:
python -m generate_dateset.py -t random -e hopper -n 1000 -o cache/hopper.pkl --render --seed 0
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


---

## Multi-Goal examples

Generate PandaReach dataset. The demonstrator needs time-feature wrapper.

```bash
python generate_dataset.py -t reach -e panda-reach-dense -n 100000 -o datasets/panda_reach_dense_100k.pkl -w time-feature
```

Generate PandaPush dataset. The demonstrator needs time-feature wrapper.

```bash
python generate_dataset.py -t tqc+her -p demonstrators/sb3_tqc_panda_push_sparse.zip -e panda-push-sparse -n 100000 -o datasets/panda_push_sparse_100k.pkl -w time-feature
```

Train Decision-Transformer on PandaReach.

```bash
python train_single.py -c configs/dt_panda_reach_dense.yaml  --dataset datasets/panda_reach_dense_100k.pkl 
```

---

## Experiments TODO

1. Train TQC on all envs.
2. Generate datasets of sizes 1m 500k 250k 100k 50k 10k.
3. Train DT on all envs and all dataset sizes.

Repeat on a different seed?