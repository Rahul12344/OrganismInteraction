#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the parent directory to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH}"

# Run the evaluator script with all arguments passed through
python "${SCRIPT_DIR}/interaction_detection_evaluator.py" "$@"