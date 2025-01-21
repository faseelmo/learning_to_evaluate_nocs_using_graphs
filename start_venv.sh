#!/bin/bash

sudo apt install -y python3.10-venv

python3.10 -m venv venv

export CUBLAS_WORKSPACE_CONFIG=:4096:8 # for torch.use_deterministic_algorithms(True)

source venv/bin/activate 
echo "Activating virtual environment..."

echo "Installing requirements..."
pip install -r requirements.txt 
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0%2Bcu121.html

snap install yq # Install yq for yaml parsing. Used in the training scripts for changing the config file.

echo "Requirements installed!"
echo "Setup complete!"

