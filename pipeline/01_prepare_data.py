"""Pipeline Step 1: Data Loading and Preprocessing.

Loads raw BCIAUT-P300 .mat files for all 15 subjects × 7 sessions,
applies minimal preprocessing, and saves 210 standardized .npz files.

Usage:
    python pipeline/01_prepare_data.py
    python pipeline/01_prepare_data.py --raw-root data/raw --processed-root data/processed
"""

from src.preprocessing.preprocess import main

if __name__ == "__main__":
    main()
