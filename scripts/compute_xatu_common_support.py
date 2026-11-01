"""Command-line entry point for run windows."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from run_windows import main


if __name__ == "__main__":
    main()
