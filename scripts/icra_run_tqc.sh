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
        ./spython.sh train_single.py -c configs/tqcher_panda_${task}_${type}_tf.yaml --seed ${SEED}
    done
done

