"""Command-line entry point for run study."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from run_study import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--mode", "real"]
    main()
