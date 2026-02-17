from pathlib import Path
import os

def get_original_cwd() -> Path:
    """Return the original working directory before Hydra changed it."""
    return Path(os.getcwd())
