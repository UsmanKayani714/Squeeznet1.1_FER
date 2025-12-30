#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to parent directory (where FERTdata is located)
cd "$PARENT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    PYTHON_CMD=".venv/bin/python3"
else
    PYTHON_CMD="python3"
fi

# Run training script from peft_extension directory
"$PYTHON_CMD" "$SCRIPT_DIR/train_peft_squeezenet.py" "$@"
