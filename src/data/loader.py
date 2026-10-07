"""Data-loading utilities for the BCIAUT-P300 dataset.

Provides the dataset inventory and session loading used by the preprocessing pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.io

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNEL_NAMES: tuple[str, ...] = ("C3", "Cz", "C4", "CPz", "P3", "Pz", "P4", "POz")
"""Fixed channel order for the BCIAUT-P300 dataset (8 centro-parietal channels)."""

DEFAULT_ACCEPTED_RAW_TIMEPOINTS: tuple[int, ...] = (350,)
"""Expected number of time-points per epoch in the raw .mat files."""

SUBJECT_IDS: tuple[str, ...] = tuple(f"SBJ{i:02d}" for i in range(1, 16))
SESSION_IDS: tuple[str, ...] = tuple(f"S{i:02d}" for i in range(1, 8))

N_CHANNELS = 8
N_CALIBRATION_EPOCHS = 1600
N_CALIBRATION_BLOCKS = 20
N_ONLINE_BLOCKS = 50
CALIBRATION_RUNS_PER_BLOCK = 10
EVENTS_PER_RUN = 8
K_RANGE = range(3, 11)  # inclusive 3..10

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionPaths:
    """Paths to all raw files for one subject × session pair."""

    subject_id: str
    session_id: str
    train_dir: Path
    test_dir: Path
    train_mat: Path
    test_mat: Path
    train_events: Path
    train_targets: Path
    train_labels: Path
    test_events: Path
    test_targets: Path
    test_labels: Path
    runs_per_block: Path


@dataclass(frozen=True)
class SessionInventoryEntry:
    """Per-session metadata collected during the inventory scan."""

    subject_id: str
    session_id: str
    paths: SessionPaths
    k_value: int
    train_mat_key: str
    test_mat_key: str
    train_format: str  # "scipy" or "hdf5"
    test_format: str


@dataclass(frozen=True)
class DatasetInventory:
    """Full inventory of all 105 sessions."""

    entries: dict[tuple[str, str], SessionInventoryEntry] = field(default_factory=dict)
    missing_files: list[str] = field(default_factory=list)
    unexpected_items: list[str] = field(default_factory=list)
    invalid_sessions: list[str] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)

    @property
    def blocking_issues(self) -> list[str]:
        """Return inventory issues that must block Step 1 processing."""
        return [*self.missing_files, *self.unexpected_items, *self.invalid_sessions]

    def is_complete(self) -> bool:
        """Return True when the raw tree matches the Step 1 contract."""
        expected_sessions = len(SUBJECT_IDS) * len(SESSION_IDS)
        return len(self.entries) == expected_sessions and not self.blocking_issues


@dataclass(frozen=True)
class MetadataRepairOperation:
    """One deterministic metadata replacement at a specific absolute index."""

    index: int
    old_value: int
    new_value: int


@dataclass(frozen=True)
class SessionMetadataRepair:
    """A session-specific metadata repair applied during loading."""

    subject_id: str
    session_id: str
    split: str
    field: str
    operations: tuple[MetadataRepairOperation, ...]
    reason: str


@dataclass(frozen=True)
class LoadedSplit:
    """Loaded data for one split (calibration or online) before preprocessing."""

    epochs: np.ndarray       # raw shape [channels, timepoints, n_epochs]
    labels: np.ndarray       # int, [n_epochs], values 0/1
    flash_ids: np.ndarray    # int, [n_epochs], values 1-8
    block_targets: np.ndarray  # int, [n_blocks], values 1-8


@dataclass(frozen=True)
class LoadedSession:
    """Loaded data for one complete subject × session."""

    subject_id: str
    session_id: str
    sub_label: str   # e.g. "sub-01"
    ses_label: str   # e.g. "ses-01"
    calibration: LoadedSplit
    online: LoadedSplit
    k_value: int
    train_mat_key: str
    test_mat_key: str
    train_format: str
    test_format: str
    repairs: tuple[SessionMetadataRepair, ...] = ()


SESSION_METADATA_REPAIRS: dict[tuple[str, str, str, str], SessionMetadataRepair] = {
    ("SBJ05", "S05", "online", "flash_ids"): SessionMetadataRepair(
        subject_id="SBJ05",
        session_id="S05",
        split="online",
        field="flash_ids",
        operations=(
            MetadataRepairOperation(index=543, old_value=6, new_value=3),
            MetadataRepairOperation(index=551, old_value=3, new_value=6),
            MetadataRepairOperation(index=559, old_value=6, new_value=3),
            MetadataRepairOperation(index=567, old_value=8, new_value=6),
            MetadataRepairOperation(index=575, old_value=3, new_value=8),
        ),
        reason=(
            "Deterministic repair for SBJ05/S05 online testEvents: five runs in "
            "block 11 contain one duplicated non-target flash and one missing "
            "flash, while the target flash remains correct."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expected_session_ids() -> list[tuple[str, str]]:
    """Return all 105 (subject_id, session_id) pairs in canonical order.

    Returns:
        List of (subject_id, session_id) tuples, e.g. [("SBJ01","S01"), ...].
    """
    return [(s, ses) for s in SUBJECT_IDS for ses in SESSION_IDS]


def build_dataset_inventory(
    raw_root: Path,
    accepted_raw_timepoints: tuple[int, ...] = DEFAULT_ACCEPTED_RAW_TIMEPOINTS,
) -> DatasetInventory:
    """Scan the raw data tree and build a full inventory (sub-step 1.1a).

    Args:
        raw_root: Path to data/raw/.
        accepted_raw_timepoints: Accepted time-axis lengths in raw MAT arrays.

    Returns:
        DatasetInventory with one entry per verified session.

    Raises:
        FileNotFoundError: If required directories are missing.
    """
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    entries: dict[tuple[str, str], SessionInventoryEntry] = {}
    missing_files: list[str] = []
    unexpected_items: list[str] = []
    invalid_sessions: list[str] = []
    deviations: list[str] = []
    expected_subjects = set(SUBJECT_IDS)
    expected_sessions = set(SESSION_IDS)

    # Check for unexpected subject folders (1.1a-02 exactness)
    for child in sorted(raw_root.iterdir()):
        if child.is_dir() and child.name not in expected_subjects:
            unexpected_items.append(f"Unexpected subject folder: {child}")

    # Check all 15 subject folders exist (1.1a-02)
    for subj_id in SUBJECT_IDS:
        subj_dir = raw_root / subj_id
        if not subj_dir.is_dir():
            missing_files.append(f"Subject folder missing: {subj_dir}")
            continue

        # Check subject contains only the 7 expected session folders (1.1a-03 exactness)
        for child in sorted(subj_dir.iterdir()):
            if child.is_dir() and child.name not in expected_sessions:
                unexpected_items.append(f"Unexpected session folder: {child}")

        # Check all 7 session folders (1.1a-03)
        for sess_id in SESSION_IDS:
            sess_dir = subj_dir / sess_id
            if not sess_dir.is_dir():
                missing_files.append(f"Session folder missing: {sess_dir}")
                continue

            # Verify Train and Test folders (1.1a-04)
            train_dir = sess_dir / "Train"
            test_dir = sess_dir / "Test"
            if not train_dir.is_dir():
                missing_files.append(f"Train folder missing: {train_dir}")
                continue
            if not test_dir.is_dir():
                missing_files.append(f"Test folder missing: {test_dir}")
                continue

            # Build paths and verify file existence (1.1a-05, 1.1a-06)
            paths = _build_session_paths(subj_id, sess_id, train_dir, test_dir)
            session_missing = _check_required_files(paths)
            if session_missing:
                missing_files.extend(session_missing)
                continue

            # Read K value (1.1a-08 through 1.1a-10)
            k_value = _read_k_value(paths.runs_per_block)
            if k_value not in K_RANGE:
                msg = f"{subj_id}/{sess_id}: K={k_value} outside range 3-10"
                invalid_sessions.append(msg)

            # Probe MAT files (1.1a-11 through 1.1a-15)
            train_key, train_fmt = _probe_mat_file(
                paths.train_mat, accepted_raw_timepoints,
            )
            test_key, test_fmt = _probe_mat_file(
                paths.test_mat, accepted_raw_timepoints,
            )

            entries[(subj_id, sess_id)] = SessionInventoryEntry(
                subject_id=subj_id,
                session_id=sess_id,
                paths=paths,
                k_value=k_value,
                train_mat_key=train_key,
                test_mat_key=test_key,
                train_format=train_fmt,
                test_format=test_fmt,
            )

    # Check key consistency across sessions (1.1a-16, 1.1a-17)
    train_keys = {e.train_mat_key for e in entries.values()}
    test_keys = {e.test_mat_key for e in entries.values()}
    if len(train_keys) > 1:
        deviations.append(f"Non-uniform train MAT keys: {train_keys}")
    if len(test_keys) > 1:
        deviations.append(f"Non-uniform test MAT keys: {test_keys}")

    if missing_files:
        for mf in missing_files:
            logger.error(mf)
    if unexpected_items:
        for issue in unexpected_items:
            logger.error(issue)
    if invalid_sessions:
        for issue in invalid_sessions:
            logger.error(issue)

    inventory = DatasetInventory(
        entries=entries,
        missing_files=missing_files,
        unexpected_items=unexpected_items,
        invalid_sessions=invalid_sessions,
        deviations=deviations,
    )
    logger.info(
        (
            "Inventory complete: %d sessions found, %d missing files, "
            "%d unexpected items, %d invalid sessions, %d deviations"
        ),
        len(entries),
        len(missing_files),
        len(unexpected_items),
        len(invalid_sessions),
        len(deviations),
    )
    return inventory


def format_dataset_inventory(inventory: DatasetInventory) -> str:
    """Format the inventory as human-readable text for dataset_inventory.txt.

    Args:
        inventory: The completed dataset inventory.

    Returns:
        Multi-line string suitable for writing to a text file.
    """
    lines: list[str] = []
    lines.append("BCIAUT-P300 Dataset Inventory")
    lines.append("=" * 50)
    lines.append(f"Sessions found: {len(inventory.entries)}")
    lines.append(f"Missing files: {len(inventory.missing_files)}")
    lines.append(f"Unexpected items: {len(inventory.unexpected_items)}")
    lines.append(f"Invalid sessions: {len(inventory.invalid_sessions)}")
    lines.append(f"Blocking issues: {len(inventory.blocking_issues)}")
    lines.append(f"Deviations: {len(inventory.deviations)}")
    lines.append("")

    # K-value table
    lines.append("K values per session:")
    lines.append("-" * 50)
    header = "Subject  " + "  ".join(SESSION_IDS)
    lines.append(header)
    for subj_id in SUBJECT_IDS:
        row = f"{subj_id}   "
        for sess_id in SESSION_IDS:
            entry = inventory.entries.get((subj_id, sess_id))
            row += f"  {entry.k_value:2d}" if entry else "   -"
        lines.append(row)
    lines.append("")

    # MAT format summary
    lines.append("MAT file format summary:")
    lines.append("-" * 50)
    for (subj_id, sess_id), entry in sorted(inventory.entries.items()):
        lines.append(
            f"  {subj_id}/{sess_id}: "
            f"train_key={entry.train_mat_key} ({entry.train_format}), "
            f"test_key={entry.test_mat_key} ({entry.test_format})"
        )
    lines.append("")

    # Missing files
    if inventory.missing_files:
        lines.append("Missing files:")
        lines.append("-" * 50)
        for mf in inventory.missing_files:
            lines.append(f"  {mf}")
        lines.append("")

    if inventory.unexpected_items:
        lines.append("Unexpected items:")
        lines.append("-" * 50)
        for item in inventory.unexpected_items:
            lines.append(f"  {item}")
        lines.append("")

    if inventory.invalid_sessions:
        lines.append("Invalid sessions:")
        lines.append("-" * 50)
        for issue in inventory.invalid_sessions:
            lines.append(f"  {issue}")
        lines.append("")

    # Deviations
    if inventory.deviations:
        lines.append("Deviations:")
        lines.append("-" * 50)
        for dev in inventory.deviations:
            lines.append(f"  {dev}")
        lines.append("")

    return "\n".join(lines)


def load_session(
    inventory: DatasetInventory,
    subject_id: str,
    session_id: str,
    accepted_raw_timepoints: tuple[int, ...] = DEFAULT_ACCEPTED_RAW_TIMEPOINTS,
) -> LoadedSession:
    """Load one subject × session pair (sub-step 1.1).

    Args:
        inventory: The dataset inventory from build_dataset_inventory.
        subject_id: e.g. "SBJ01".
        session_id: e.g. "S01".
        accepted_raw_timepoints: Accepted time-axis lengths in raw MAT arrays.

    Returns:
        LoadedSession with calibration and online splits.

    Raises:
        KeyError: If the session is not in the inventory.
        ValueError: If loaded data fails shape or range validation.
    """
    entry = inventory.entries.get((subject_id, session_id))
    if entry is None:
        raise KeyError(
            f"Session {subject_id}/{session_id} not found in inventory"
        )

    paths = entry.paths
    k = entry.k_value
    n_online_epochs = N_ONLINE_BLOCKS * EVENTS_PER_RUN * k  # 400 * K
    applied_repairs: list[SessionMetadataRepair] = []

    # Load calibration (1.1-01 through 1.1-07)
    cal_epochs = _load_mat_array(
        paths.train_mat, entry.train_mat_key, entry.train_format,
    )
    _validate_raw_shape(
        cal_epochs, N_CHANNELS, accepted_raw_timepoints, N_CALIBRATION_EPOCHS,
        f"{subject_id}/{session_id}/Train",
    )
    cal_targets = _load_text_file(paths.train_targets, np.int8)
    cal_events = _load_text_file(paths.train_events, np.int8)
    cal_block_targets = _load_text_file(paths.train_labels, np.int8)

    _validate_label_array(cal_targets, N_CALIBRATION_EPOCHS, {0, 1}, "trainTargets")
    _validate_label_array(cal_events, N_CALIBRATION_EPOCHS, set(range(1, 9)), "trainEvents")
    _validate_label_array(cal_block_targets, N_CALIBRATION_BLOCKS, set(range(1, 9)), "trainLabels")

    # Load online (1.1-08 through 1.1-14)
    online_epochs = _load_mat_array(
        paths.test_mat, entry.test_mat_key, entry.test_format,
    )
    _validate_raw_shape(
        online_epochs, N_CHANNELS, accepted_raw_timepoints, n_online_epochs,
        f"{subject_id}/{session_id}/Test",
    )
    online_targets = _load_text_file(paths.test_targets, np.int8)
    online_events = _load_text_file(paths.test_events, np.int8)
    online_block_targets = _load_text_file(paths.test_labels, np.int8)

    _validate_label_array(online_targets, n_online_epochs, {0, 1}, "testTargets")
    _validate_label_array(online_events, n_online_epochs, set(range(1, 9)), "testEvents")
    _validate_label_array(online_block_targets, N_ONLINE_BLOCKS, set(range(1, 9)), "testLabels")
    online_events, repairs = _apply_registered_metadata_repairs(
        online_events,
        subject_id=subject_id,
        session_id=session_id,
        split="online",
        field="flash_ids",
    )
    if repairs:
        applied_repairs.extend(repairs)
        _validate_label_array(
            online_events, n_online_epochs, set(range(1, 9)), "testEvents (repaired)",
        )

    # Build BIDS-style labels
    subj_num = int(subject_id.replace("SBJ", ""))
    sess_num = int(session_id.replace("S", ""))
    sub_label = f"sub-{subj_num:02d}"
    ses_label = f"ses-{sess_num:02d}"

    return LoadedSession(
        subject_id=subject_id,
        session_id=session_id,
        sub_label=sub_label,
        ses_label=ses_label,
        calibration=LoadedSplit(
            epochs=cal_epochs,
            labels=cal_targets,
            flash_ids=cal_events,
            block_targets=cal_block_targets,
        ),
        online=LoadedSplit(
            epochs=online_epochs,
            labels=online_targets,
            flash_ids=online_events,
            block_targets=online_block_targets,
        ),
        k_value=k,
        train_mat_key=entry.train_mat_key,
        test_mat_key=entry.test_mat_key,
        train_format=entry.train_format,
        test_format=entry.test_format,
        repairs=tuple(applied_repairs),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_session_paths(
    subj_id: str, sess_id: str, train_dir: Path, test_dir: Path,
) -> SessionPaths:
    """Build all file paths for one session."""
    return SessionPaths(
        subject_id=subj_id,
        session_id=sess_id,
        train_dir=train_dir,
        test_dir=test_dir,
        train_mat=train_dir / "trainData.mat",
        test_mat=test_dir / "testData.mat",
        train_events=train_dir / "trainEvents.txt",
        train_targets=train_dir / "trainTargets.txt",
        train_labels=train_dir / "trainLabels.txt",
        test_events=test_dir / "testEvents.txt",
        test_targets=test_dir / "testTargets.txt",
        test_labels=test_dir / "testLabels.txt",
        runs_per_block=test_dir / "runs_per_block.txt",
    )


def _apply_registered_metadata_repairs(
    values: np.ndarray,
    *,
    subject_id: str,
    session_id: str,
    split: str,
    field: str,
) -> tuple[np.ndarray, tuple[SessionMetadataRepair, ...]]:
    """Apply any deterministic repair registered for one metadata vector."""
    repair = SESSION_METADATA_REPAIRS.get((subject_id, session_id, split, field))
    if repair is None:
        return values, ()

    repaired = values.copy()
    for op in repair.operations:
        if op.index < 0 or op.index >= repaired.shape[0]:
            raise ValueError(
                f"{subject_id}/{session_id}/{split}/{field}: repair index {op.index} out of bounds"
            )
        observed = int(repaired[op.index])
        if observed != op.old_value:
            raise ValueError(
                (
                    f"{subject_id}/{session_id}/{split}/{field}: repair expected "
                    f"value {op.old_value} at index {op.index}, got {observed}"
                )
            )
        repaired[op.index] = op.new_value

    logger.warning(
        "Applied metadata repair for %s/%s %s %s (%d replacements)",
        subject_id,
        session_id,
        split,
        field,
        len(repair.operations),
    )
    return repaired, (repair,)


def _check_required_files(paths: SessionPaths) -> list[str]:
    """Return list of missing-file messages for a session."""
    missing: list[str] = []
    required = [
        paths.train_mat, paths.train_events, paths.train_targets, paths.train_labels,
        paths.test_mat, paths.test_events, paths.test_targets, paths.test_labels,
        paths.runs_per_block,
    ]
    for f in required:
        if not f.is_file():
            missing.append(f"File missing: {f}")
    return missing


def _read_k_value(path: Path) -> int:
    """Read the K value (runs per block) from runs_per_block.txt."""
    text = path.read_text(encoding="utf-8").strip()
    return int(text)


def _probe_mat_file(
    path: Path,
    accepted_raw_timepoints: Sequence[int],
) -> tuple[str, str]:
    """Determine the EEG array key and format of a .mat file.

    Tries scipy.io.whosmat first; falls back to mat73 for HDF5/v7.3 files.

    Args:
        path: Path to the .mat file.
        accepted_raw_timepoints: Acceptable time-axis lengths.

    Returns:
        (key_name, format_string) where format_string is "scipy" or "hdf5".
    """
    try:
        variables = scipy.io.whosmat(str(path))
        for name, shape, _ in variables:
            if len(shape) == 3 and shape[0] == N_CHANNELS and shape[1] in accepted_raw_timepoints:
                return name, "scipy"
        # No matching 3-D array found; try all variables
        for name, shape, _ in variables:
            if len(shape) == 3:
                return name, "scipy"
        raise ValueError(
            f"No 3-D EEG array found in {path}. Variables: {variables}"
        )
    except NotImplementedError:
        # HDF5 / MATLAB v7.3 format
        return _probe_mat_file_hdf5(path, accepted_raw_timepoints)


def _probe_mat_file_hdf5(
    path: Path,
    accepted_raw_timepoints: Sequence[int],
) -> tuple[str, str]:
    """Probe an HDF5 .mat file using mat73."""
    try:
        import mat73  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "mat73 is required for HDF5 .mat files. Install with: pip install mat73"
        ) from exc

    data = mat73.loadmat(str(path))
    for name, arr in data.items():
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            if arr.shape[0] == N_CHANNELS and arr.shape[1] in accepted_raw_timepoints:
                return name, "hdf5"
    # Fallback: return first 3-D array
    for name, arr in data.items():
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            return name, "hdf5"
    raise ValueError(f"No 3-D EEG array found in HDF5 file {path}")


def _load_mat_array(path: Path, key: str, fmt: str) -> np.ndarray:
    """Load the EEG array from a .mat file.

    Args:
        path: Path to the .mat file.
        key: Variable name inside the file.
        fmt: "scipy" or "hdf5".

    Returns:
        Raw EEG array with shape [channels, timepoints, n_epochs].
    """
    if fmt == "scipy":
        mat = scipy.io.loadmat(str(path))
        return np.asarray(mat[key], dtype=np.float64)
    elif fmt == "hdf5":
        try:
            import mat73  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "mat73 is required for HDF5 .mat files."
            ) from exc
        data = mat73.loadmat(str(path))
        return np.asarray(data[key], dtype=np.float64)
    else:
        raise ValueError(f"Unknown MAT format: {fmt!r}")


def _load_text_file(path: Path, dtype: type = np.int8) -> np.ndarray:
    """Load a single-column text file as a 1-D numpy array.

    Args:
        path: Path to the text file (one integer per line).
        dtype: Target dtype.

    Returns:
        1-D array of the specified dtype.
    """
    return np.loadtxt(str(path), dtype=dtype)


def _validate_raw_shape(
    arr: np.ndarray,
    expected_channels: int,
    accepted_timepoints: Sequence[int],
    expected_epochs: int,
    context: str,
) -> None:
    """Validate the shape of a raw EEG array [channels, timepoints, epochs]."""
    if arr.ndim != 3:
        raise ValueError(f"{context}: expected 3-D array, got {arr.ndim}-D")
    c, t, n = arr.shape
    if c != expected_channels:
        raise ValueError(f"{context}: expected {expected_channels} channels, got {c}")
    if t not in accepted_timepoints:
        raise ValueError(
            f"{context}: timepoints {t} not in accepted set {accepted_timepoints}"
        )
    if n != expected_epochs:
        raise ValueError(f"{context}: expected {expected_epochs} epochs, got {n}")


def _validate_label_array(
    arr: np.ndarray,
    expected_length: int,
    valid_values: set[int],
    context: str,
) -> None:
    """Validate a 1-D label array."""
    if arr.ndim != 1:
        raise ValueError(f"{context}: expected 1-D array, got {arr.ndim}-D")
    if arr.shape[0] != expected_length:
        raise ValueError(
            f"{context}: expected length {expected_length}, got {arr.shape[0]}"
        )
    unique = set(arr.tolist())
    if not unique.issubset(valid_values):
        raise ValueError(
            f"{context}: unexpected values {unique - valid_values}; "
            f"expected subset of {valid_values}"
        )
