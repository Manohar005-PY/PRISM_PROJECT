"""Dataset and run loading engine for PRISM Part 17 evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class DatasetLoadError(Exception):
    """Raised when dataset files cannot be found or parsed."""


class DatasetLoader:
    """Loads Part 16 dataset manifests, metadata, and JSONL streams."""

    def __init__(self, dataset_dir: Path | str) -> None:
        self.dataset_dir: Path = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise DatasetLoadError(f"Dataset directory not found: {self.dataset_dir}")

    def load_manifest(self) -> dict[str, Any]:
        """Load and parse dataset_manifest.json."""
        manifest_path = self.dataset_dir / "dataset_manifest.json"
        if not manifest_path.exists():
            raise DatasetLoadError(f"Missing dataset_manifest.json in {self.dataset_dir}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def iter_runs(
        self,
        scenario_family_filter: str | None = None,
        run_id_filter: str | None = None,
        noise_filter: str | None = None,
    ) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
        """Iterate over matching runs and load their metadata and time-series streams.

        Yields:
            tuple (metadata, ground_truth_records, observation_records, prism_output_records)
        """
        manifest = self.load_manifest()

        for sc in manifest["scenarios"]:
            sc_id = sc["scenario_id"]
            sc_fam = sc["scenario_family"]
            rel_dir = sc["run_dir"]

            if scenario_family_filter and sc_fam != scenario_family_filter:
                continue
            if run_id_filter and sc_id != run_id_filter:
                continue

            run_dir = self.dataset_dir / rel_dir
            if not run_dir.exists():
                raise DatasetLoadError(f"Run directory not found on disk: {run_dir}")

            metadata, gt_recs, obs_recs, pr_recs = self.load_run(run_dir)

            # Check noise condition filter
            noise_cfg = metadata.get("noise_config", {})
            noise_level = metadata.get("parameters", {}).get("noise_level", "NO_NOISE")
            if noise_filter and noise_level != noise_filter:
                continue

            yield metadata, gt_recs, obs_recs, pr_recs

    def load_run(
        self,
        run_dir: Path | str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Load an individual run directory."""
        run_path = Path(run_dir)

        meta_file = run_path / "metadata.json"
        gt_file = run_path / "ground_truth.jsonl"
        obs_file = run_path / "observations.jsonl"
        prism_file = run_path / "prism_output.jsonl"

        for f in (meta_file, gt_file, obs_file, prism_file):
            if not f.exists():
                raise DatasetLoadError(f"Required run file missing: {f}")

        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        gt_records = self._read_jsonl(gt_file)
        obs_records = self._read_jsonl(obs_file)
        prism_records = self._read_jsonl(prism_file)

        return metadata, gt_records, obs_records, prism_records

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Read and parse all JSON lines from a file."""
        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean = line.strip()
                if clean:
                    try:
                        records.append(json.loads(clean))
                    except json.JSONDecodeError as e:
                        raise DatasetLoadError(f"JSON decode error in {path} at line {line_idx}: {e}")
        return records
