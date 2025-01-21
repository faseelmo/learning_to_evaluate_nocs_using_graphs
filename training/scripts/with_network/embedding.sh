#!/bin/bash

# File paths
YAML_FILE="training/config/params_with_network.yaml"  # Path to your YAML file
TRAIN_SCRIPT="python3 -m training.train_with_network"  # Command to run training
UPDATE_SCRIPT="./training/scripts/update_yaml.sh"  # Path to your update script

$UPDATE_SCRIPT "$YAML_FILE" RESULTS_DIR "training/results/with_network/1_embedding_m3_c"

embedding_list=(1 2 4 8 16 32 64 128)
# embedding_list=(64 128)

for embedding in "${embedding_list[@]}"; do

    echo "Training for embedding: $embedding"
    RUN_DIR="E${embedding}"
    $UPDATE_SCRIPT "$YAML_FILE" EMBEDDING_DIM "$embedding"

    echo "Training for embedding: $embedding"
    $TRAIN_SCRIPT "$RUN_DIR"

done