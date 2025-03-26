#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the project root directory to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Check if conda environment exists
if ! conda env list | grep -q "^organismtraining "; then
    echo "Creating conda environment 'organismtraining'..."
    conda create -n organismtraining python=3.10 -y
    conda activate organismtraining
    echo "Installing requirements..."
    python3 -m pip install -r requirements.txt
else
    echo "Conda environment 'organismtraining' already exists."
    conda activate organismtraining
fi