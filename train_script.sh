#!/bin/bash

args=(
    "interaction.novelty_mode='state'"
    "interaction.novelty_mode='latent'"
)

output_folders=(
    "novel_state_300k"
    "novel_latent_300k"
)

len=${#args[@]}

for ((i=0; i<$len; i++)); do
    # Do multiple runs for each arg config
    for j in {1..1}; do
        output_folder="${output_folders[$i]}_$j"
        ./online/run_gcrl.sh env.name='MountainCar-v0' ${args[$i]} output_folder=$output_folder
    done
done