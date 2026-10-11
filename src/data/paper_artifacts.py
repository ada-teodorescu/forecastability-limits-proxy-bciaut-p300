"""Locate saved paper tables supplied as CSV or lossless gzip archives."""

from pathlib import Path


def resolve_artifact(path: str | Path) -> Path:
    """Prefer a freshly generated CSV; otherwise use its shipped .csv.gz file."""
    path = Path(path)
    if path.is_file():
        return path
    compressed = path.with_suffix(path.suffix + ".gz")
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(f"Missing artifact: {path} (or {compressed})")
