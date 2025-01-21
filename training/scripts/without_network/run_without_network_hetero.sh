#!/bin/bash

# Path to the main script
MAIN_SCRIPT="./training/scripts/run_without_network_experiments.sh"

# Define the combinations
combinations=(
    "--is_hetero True"
    "--is_hetero True --has_scheduler True"
    "--is_hetero True --has_dependency True"
    "--is_hetero True --has_dependency True --has_scheduler True"
    "--is_hetero True --has_dependency True --has_exit True"
    "--is_hetero True --has_dependency True --has_exit True --has_scheduler True"
)

# Iterate over the combinations and call the main script
for combination in "${combinations[@]}"; do
    echo "Running with combination: $combination"
    bash $MAIN_SCRIPT $combination
done