"""Command-line entry point for run contributions."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from run_contributions import main


if __name__ == "__main__":
    main()
