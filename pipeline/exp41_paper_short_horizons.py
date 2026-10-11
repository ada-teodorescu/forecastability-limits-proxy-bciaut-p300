"""Run the paper's five-model extension at horizons 1--4."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.exp41_paper_short_horizons import main


if __name__ == "__main__":
    main()
