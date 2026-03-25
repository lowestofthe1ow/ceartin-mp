import subprocess
import sys
from pathlib import Path

import questionary

# Define the available datasets for finetuning
DATASETS = ["tatoeba", "newsph-nli", "combined", "combined-stress"]
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


# Prompt user to select a training set
selected_dataset = questionary.select(
    "Select a dataset for finetuning:", choices=DATASETS, use_indicator=True
).ask()

if not selected_dataset:
    print("Cancelled by user.")
    sys.exit(0)

# Ask if the user wants to resume from a checkpoint
resume_path = None
if questionary.confirm("Resume from a checkpoint?", default=False).ask():
    available_checkpoints = get_checkpoints()
    if available_checkpoints:
        resume_path = questionary.select(
            "Select checkpoint to resume from:", choices=available_checkpoints
        ).ask()
    else:
        print("No checkpoints found. Starting a fresh run.")

# Prompt user for a run description
description = questionary.text("Enter a description for this run:").ask()

if description is None:
    print("Cancelled by user.")
    sys.exit(0)

# NOTE: We run the Python script with uv!
cmd = [
    "uv",
    "run",
    "python3",
    "-m",
    "src.scripts.g2p_finetune",
    "--dataset",
    selected_dataset,
    "--description",
    description,
]

# Add resume argument if a checkpoint was selected
if resume_path:
    cmd.extend(["--resume-from", resume_path])

print(f"\nRunning command: {' '.join(cmd)}\n")

subprocess.run(cmd, check=True)
