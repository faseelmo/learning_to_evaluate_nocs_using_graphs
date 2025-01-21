#!/bin/bash

# Usage: ./update_yaml.sh <yaml_file> <parameter> <value>
# Example: ./update_yaml.sh config.yaml NUM_MPN_LAYERS 5


if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <yaml_file> <parameter> <value>"
    exit 1
fi

YAML_FILE="$1"
PARAMETER="$2"
NEW_VALUE="$3"

# Check if the file exists
if [ ! -f "$YAML_FILE" ]; then
    echo "Error: YAML file '$YAML_FILE' not found!"
    exit 1
fi

# Check if NEW_VALUE is numeric or string and handle accordingly
if [[ "$NEW_VALUE" =~ ^[0-9]+$ || "$NEW_VALUE" =~ ^[0-9]+\.[0-9]+$ ]]; then
    # Numeric value
    yq e ".${PARAMETER} = ${NEW_VALUE}" -i "$YAML_FILE"
else
    # String value
    yq e ".${PARAMETER} = \"${NEW_VALUE}\"" -i "$YAML_FILE"
fi

# Check if the update was successful
if [ $? -eq 0 ]; then
    echo "Updated $PARAMETER to $NEW_VALUE in $YAML_FILE successfully!"
else
    echo "Failed to update $PARAMETER in $YAML_FILE."
fi