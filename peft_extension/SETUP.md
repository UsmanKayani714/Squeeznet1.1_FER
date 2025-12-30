# Setup Instructions

## Python Command
On macOS, use `python3` instead of `python`.

## Install Dependencies

If PyTorch is not installed, install it first:

```bash
# Activate virtual environment (if using one)
source ../.venv/bin/activate

# Install PyTorch with MPS support (for Apple Silicon)
pip install torch torchvision

# Or if you need specific version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Running the Scripts

All scripts should be run with `python3`:

```bash
# Train PEFT model
python3 train_peft_squeezenet.py --bs 32 --lr 0.001

# Evaluate PEFT model
python3 evaluate_peft.py

# Compare models
python3 compare_models.py

# Test model structure
python3 test_model.py
```

## Alternative: Create an alias

You can add this to your `~/.zshrc` to use `python` as an alias for `python3`:

```bash
alias python=python3
```

Then reload your shell:
```bash
source ~/.zshrc
```

