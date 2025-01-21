#!/bin/bash

# File paths
YAML_FILE="training/config/params_with_network.yaml"  # Path to your YAML file
TRAIN_SCRIPT="python3 -m training.train_with_network"  # Command to run training
UPDATE_SCRIPT="./training/scripts/update_yaml.sh"  # Path to your update script
UPDATE_LIST_SCRIPT="./training/scripts/with_network/update_list_in_yaml.sh"  # Path to your update list script

$UPDATE_SCRIPT "$YAML_FILE" RESULTS_DIR "training/results/with_network/3_aggr_hetero"

# Original list
original_list=("sum" "mean" "max" )

# Function to generate subsets
generate_subsets() {
  local list=("$@")
  local subsets=()
  local n=${#list[@]}
  for ((i=1; i<1<<n; i++)); do
    subset=()
    for ((j=0; j<n; j++)); do
      if ((i & (1 << j))); then
        subset+=("${list[j]}")
      fi
    done
    subsets+=("$(IFS=,; echo "${subset[*]}")")
  done
  echo "${subsets[@]}"
}

# Generate all subsets
subsets=$(generate_subsets "${original_list[@]}")

# Use each subset with the update script
for subset in $subsets; do
  # Split the subset into an array
  IFS=',' read -ra subset_array <<< "$subset"
  
  # Call the update script with the current subset
  $UPDATE_LIST_SCRIPT "${subset_array[@]}"
  
  RUN_DIR=$(IFS=_; echo "${subset_array[*]}")

  $TRAIN_SCRIPT "$RUN_DIR"

done
