#!/bin/bash

# Check if at least one argument is passed
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <value1> [value2 ...]"
  exit 1
fi

# Define the YAML file
yaml_file="training/config/params_with_network.yaml"  # Replace with your YAML file name

# Collect all arguments into an array
values=("$@")

# Build the YAML array format
yaml_array=$(printf ', "%s"' "${values[@]}")
yaml_array="[${yaml_array:2}]"  # Remove the leading comma and space

# Update the YAML file using yq
yq -i ".HETERO_AGGR = $yaml_array" "$yaml_file"

echo "Updated HETERO_AGGR to ${yaml_array} in $yaml_file"

