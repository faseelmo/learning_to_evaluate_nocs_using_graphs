#!/bin/bash

# File paths
YAML_FILE="training/config/params_without_network.yaml"  # Path to your YAML file
TRAIN_SCRIPT="python3 -m training.train_without_network"  # Command to run training
UPDATE_SCRIPT="./training/scripts/update_yaml.sh"  # Path to your update script

# Parameters to iterate over
CONV_TYPES=("graphconv") # "graphconv" "gcn" "gin" "sage" "gat"
NUM_LAYERS=(5)
HIDDEN_CHANNELS=(64)
LOSS_FUNCTIONS=("mae") # "mse" "mae" "huber"
AGGR=("add") # "mean" "max" "min" "sum"
MAX_EPOCHS=200

IS_HETERO=False
HAS_SCHEDULER=False 
HAS_DEPENDENCY=False
HAS_TASK_DEPEND=False

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --is_hetero) IS_HETERO="$2"; shift ;;
        --has_scheduler) HAS_SCHEDULER="$2"; shift ;;
        --has_dependency) HAS_DEPENDENCY="$2"; shift ;;
        --has_exit) HAS_EXIT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

$UPDATE_SCRIPT "$YAML_FILE" EPOCHS "$MAX_EPOCHS"
$UPDATE_SCRIPT "$YAML_FILE" IS_HETERO "$IS_HETERO"
$UPDATE_SCRIPT "$YAML_FILE" HAS_EXIT "$HAS_EXIT"
$UPDATE_SCRIPT "$YAML_FILE" HAS_DEPENDENCY "$HAS_DEPENDENCY"
$UPDATE_SCRIPT "$YAML_FILE" HAS_SCHEDULER "$HAS_SCHEDULER"

# Iterate over all parameter combinations
for CONV in "${CONV_TYPES[@]}"; do
    for LAYER in "${NUM_LAYERS[@]}"; do
        for CHANNEL in "${HIDDEN_CHANNELS[@]}"; do
            for LOSS in "${LOSS_FUNCTIONS[@]}"; do
                for AGGR in "${AGGR[@]}"; do

                    # Update parameters in the YAML file using update_yaml.sh
                    $UPDATE_SCRIPT "$YAML_FILE" CONV_TYPE "$CONV"
                    $UPDATE_SCRIPT "$YAML_FILE" NUM_MPN_LAYERS "$LAYER"
                    $UPDATE_SCRIPT "$YAML_FILE" HIDDEN_CHANNELS "$CHANNEL"
                    $UPDATE_SCRIPT "$YAML_FILE" AGGR "$AGGR"
                    $UPDATE_SCRIPT "$YAML_FILE" LOSS_FN "$LOSS"

                    # Generate a unique directory for this run
                    RUN_DIR="${CONV}_L${LAYER}_C${CHANNEL}_A${AGGR}_${LOSS}"

                    if [ "$IS_HETERO" = True ]; then
                        RUN_DIR="${RUN_DIR}_HETERO"
                    fi

                    if [ "$HAS_EXIT" = True ]; then
                        RUN_DIR="${RUN_DIR}_EXIT"
                    fi

                    if [ "$HAS_SCHEDULER" = True ]; then
                        RUN_DIR="${RUN_DIR}_SCHEDULER"
                    fi

                    if [ "$HAS_DEPENDENCY" = True ]; then
                        RUN_DIR="${RUN_DIR}_DEPENDENCY"
                    fi

                    # Run training
                    echo "Running training for CONV=$CONV, LAYERS=$LAYER, CHANNELS=$CHANNEL, AGGR=$AGGR, LOSS=$LOSS, HETERO=$IS_HETERO, DEPENDENCY=$HAS_DEPENDENCY, EXIT=$HAS_EXIT, SCHEDULER=$HAS_SCHEDULER"
                    $TRAIN_SCRIPT "$RUN_DIR"

                    # Check if training succeeded
                    if [ $? -ne 0 ]; then
                        echo "Training failed for CONV=$CONV, LAYERS=$LAYER, CHANNELS=$CHANNEL, AGGR=$AGGR, LOSS=$LOSS, HETERO=$IS_HETERO, DEPENDENCY=$HAS_DEPENDENCY, EXIT=$HAS_EXIT, SCHEDULER=$HAS_SCHEDULER"
                    else
                        echo "Training succeeded for CONV=$CONV, LAYERS=$LAYER, CHANNELS=$CHANNEL, AGGR=$AGGR, LOSS=$LOSS, HETERO=$IS_HETERO, DEPENDENCY=$HAS_DEPENDENCY, EXIT=$HAS_EXIT, SCHEDULER=$HAS_SCHEDULER"
                    fi

                done
            done
        done
    done
done

echo "All experiments completed!"
