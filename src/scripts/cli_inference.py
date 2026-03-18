import subprocess
import sys
from pathlib import Path

import questionary

DEFAULT_CHECKPOINTS_PATH = "models/checkpoints"


def get_checkpoints(base_dir=DEFAULT_CHECKPOINTS_PATH):
    path = Path(base_dir)
    if not path.exists():
        print(f"Error: Directory '{base_dir}' does not exist.")
        sys.exit(1)

    # Recursively check all checkpoint subdirectories
    checkpoints = [
        str(p) for p in path.glob("*/*") if p.is_dir() and "checkpoint" in p.name
    ]

    return sorted(checkpoints)


# List all available checkpoints
checkpoints = get_checkpoints()
if not checkpoints:
    print("No checkpoints found.")
    sys.exit(1)

# Prompt user to select a checkpoint
selected_checkpoint = questionary.select(
    "Choose a checkpoint for inference:", choices=checkpoints, use_indicator=True
).ask()

if not selected_checkpoint:
    print("Cancelled by user.")
    sys.exit(0)

# Prompt user to enter a sentence
sentence = questionary.text("Enter the sentence to process:").ask()

if not sentence:
    print("No sentence provided. Exiting.")
    sys.exit(0)

# NOTE: We run the Python script with uv!
cmd = [
    "uv",
    "run",
    "python3",
    "-m",
    "src.scripts.g2p_inference",
    sentence,
    "--checkpoint-path",
    selected_checkpoint,
]

print(f"\nRunning command: {' '.join(cmd)}\n")

subprocess.run(cmd, check=True)
