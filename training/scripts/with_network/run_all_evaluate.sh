#!/bin/bash

# Root directory containing the experiments
ROOT_DIR="training/results/with_network/aggr_conv"

# Python script and argument
PYTHON_SCRIPT="python3 -m training.evaluate"
ARG="--with_network"

echo "Directories in $path:"
for subdir in "$ROOT_DIR"/*/; do
    # Check if the path is a directory (handles cases where no directories exist)
    if [ -d "$subdir" ]; then
        echo "$subdir"
        $PYTHON_SCRIPT --model_path "$subdir" $ARG
        echo " "
    fi
done