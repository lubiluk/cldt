#!/bin/bash

# Check if seed is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

# Seed from the command line argument
SEED="$1"

for task in reach push pick_and_place
do
    for type in sparse dense
    do
        for ratio in 1_0 0_75 0_5 0_25 0_0
        do 
            ./spython.sh train_single.py -c configs/dt_panda_${task}_${type}_tf.yaml --dataset datasets/panda_${task}_${type}_1m_expert_ratio_${ratio}.pkl --save-path trained/dt_panda_${task}_${type}_1m_expert_ratio_${ratio} --seed ${SEED}
        done
    done
done

