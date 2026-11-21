"""Portable data/artifact paths with support for the original local workspace."""
import os
from pathlib import Path


def study_paths(repo_root):
    repo_root = Path(repo_root).resolve()
    local_data = repo_root.parent / 'data'
    has_legacy_data = (local_data / 'CIC2017_improved.parquet').is_file()
    default_data = local_data if has_legacy_data else repo_root / 'data'
    data = Path(os.environ.get('NIDS_DATA_DIR', default_data)).expanduser().resolve()
    legacy_artifacts = repo_root.parent / 'binary_baseline_results'
    default_artifacts = repo_root.parent if has_legacy_data and legacy_artifacts.is_dir() else repo_root / 'artifacts'
    artifacts = Path(os.environ.get('NIDS_ARTIFACT_DIR', default_artifacts)).expanduser().resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    return data, artifacts
