
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
python generate_dataset.py -t reach -e panda-reach-dense -n 100000 -o datasets/panda_reach_dense_random.pkl -w time-feature
python generate_dataset.py -t reach -p demonstrators/tqcher_panda_reach_dense_tf.zip -e panda-reach-dense -n 100000 -o datasets/panda_reach_dense_expert.pkl -w time-feature
```

Generate PandaPush dataset. The demonstrator needs time-feature wrapper.

```bash
python generate_dataset.py -t tqc+her -p demonstrators/sb3_tqc_panda_push_sparse.zip -e panda-push-sparse -n 100000 -o datasets/panda_push_sparse_100k_expert.pkl -w time-feature
```

Train Decision-Transformer on PandaReach.

```bash
python train_single.py -c configs/dt_panda_reach_dense.yaml  --dataset /net/tscratch/people/plgdomin088/datasets/panda_reach_dense_100k.pkl 
```

---

## Experiments TODO

1. Train TQC on all envs.
2. Generate datasets of sizes 1m 500k 250k 100k 50k 10k.
3. Train DT on all envs and all dataset sizes.

Repeat on a different seed?

--- 

## Train fast using RL Zoo

```bash
python -m rl_zoo3.train --env PandaPushDense-v3 --algo tqc --conf-file configs/tqcher_zoo.yaml --folder trained --save-freq 100000 --hyperparams n_envs:4 gradient_steps:-1
```

`n_envs` tells how many environments should work in parallel.
