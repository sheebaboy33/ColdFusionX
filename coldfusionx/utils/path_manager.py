import os
from pathlib import Path

# Navigates up 3 levels: utils/ -> coldfusionx/ -> Project Root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def get_file_path(category, filename):
    paths = {
        "metrics": BASE_DIR / "metrics" / filename,
        "data": BASE_DIR / "data" / filename,
        "artifacts": BASE_DIR / "artifacts" / "release_demo" / filename
    }
    return str(paths.get(category))