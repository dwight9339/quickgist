#!/bin/bash

# Update package lists and install required dependencies
apt-get update
apt-get install -y python3 python3-pip python3-venv

# Create a Python virtual environment
python3 -m venv distilbert_env

# Activate the virtual environment
source distilbert_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU-only version)
pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install Hugging Face's Transformers library
pip install transformers

# Print success message and instructions
echo "DistilBERT setup complete."
echo "To use the environment, run 'source distilbert_env/bin/activate'."
