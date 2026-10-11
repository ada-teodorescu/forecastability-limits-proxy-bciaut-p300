"""Create or verify hashes for the portable paper reproduction release."""

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/paper_revision/release_manifest.json"
VERIFICATION = ROOT / "docs/paper_revision/release_verification.json"
SHORT = ROOT / "data/forecasting/review_revision/short_horizons"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_manifest():
    # Include tracked files and new files allowed by .gitignore. This excludes
    # local caches, upstream EEG, checkpoints, and private writing materials.
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True,
    ).splitlines()
    excluded = {MANIFEST.relative_to(ROOT).as_posix(), VERIFICATION.relative_to(ROOT).as_posix()}
    entries = []
    for name in sorted(set(names) - excluded):
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = {"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
        if name.endswith(".csv.gz"):
            result = hashlib.sha256()
            with gzip.open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    result.update(chunk)
            entry["uncompressed_sha256"] = result.hexdigest()
        entries.append(entry)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "release_date": "2026-10-11",
        "paper_version": "Forecastability Limits of a P300-Derived EEG Proxy in BCIAUTP300-8.pdf",
        "historical_gru_source": "Thesis commit 48de5b0, preserved in src/experiments/exp41_historical_step8.py",
        "paper_training": {"max_epochs": 200, "patience": 20, "gradient_clipping": False},
        "main_models_retrained_for_release": False,
        "gzip_policy": "Lossless CSV compression; uncompressed hashes identify the original numerical artifacts.",
        "historical_provenance": "Original thesis manifests are retained under review_revision/provenance and short_horizons/provenance; their upstream EEG/private manuscript paths describe historical runs and are not a list of shipped files.",
        "files": entries,
    }, indent=2) + "\n", encoding="utf-8")


def refresh_short_manifest():
    paths = [path for path in SHORT.rglob("*") if path.is_file()
             and "checkpoints" not in path.parts and path.name != "deliverable_sha256.json"]
    paths += [ROOT / name for name in [
        "src/experiments/exp41_paper_short_horizons.py", "src/experiments/exp41_historical_step8.py",
        "pipeline/exp41_paper_short_horizons.py", "pipeline/verify_short_horizons.py",
        "tests/test_exp41_paper_short_horizons.py", "docs/paper_revision/short_horizon_protocol.md",
        "docs/paper_revision/short_horizon_results.md",
        "outputs/forecasting/review_revision/short_horizons/short_horizon_mae.png",
    ]]
    (SHORT / "deliverable_sha256.json").write_text(json.dumps({
        path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)
    }, indent=2) + "\n", encoding="utf-8")


def verify():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), f"Missing release file: {path}"
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], path
        assert path.stat().st_size < 100_000_000, f"File exceeds the release size limit: {path}"
        if "uncompressed_sha256" in item:
            result = hashlib.sha256()
            with gzip.open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    result.update(chunk)
            assert result.hexdigest() == item["uncompressed_sha256"], path
    short = json.loads((SHORT / "deliverable_sha256.json").read_text(encoding="utf-8"))
    for name, checksum in short.items():
        assert digest(ROOT / name) == checksum, name
    report = {"passed": True, "release_files_verified": len(manifest["files"]),
              "short_horizon_artifacts_verified": len(short),
              "largest_file_bytes": max(item["bytes"] for item in manifest["files"]),
              "models_retrained": False}
    VERIFICATION.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--refresh-short-manifest", action="store_true")
    args = parser.parse_args()
    if args.refresh_short_manifest:
        refresh_short_manifest()
    if args.write_manifest:
        write_manifest()
    if not args.refresh_short_manifest or args.write_manifest:
        verify()


if __name__ == "__main__":
    main()
