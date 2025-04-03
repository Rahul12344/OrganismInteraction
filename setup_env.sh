#!/bin/bash

# Get the absolute path of the parent directory
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# Print confirmation
echo "Added $PROJECT_ROOT to PYTHONPATH"
echo "Current PYTHONPATH: $PYTHONPATH"

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

# Optional: Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

echo "Setup complete! You can now run the model with:"
echo "python -m organismtraining.interaction_detection_evaluator_dropout --dataset_path <path> --model_path <path> --retrain"