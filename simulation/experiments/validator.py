"""Dataset validation and integrity checking engine for PRISM Part 16."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class DatasetValidationError(Exception):
    """Raised when a generated synthetic dataset fails integrity or schema checks."""


class DatasetValidator:
    """Performs rigorous integrity, schema, and scientific consistency checks on generated datasets."""

    def __init__(self, dataset_dir: Path | str) -> None:
        self.dataset_dir: Path = Path(dataset_dir)

    def validate_dataset(self) -> dict[str, Any]:
        """Validate entire dataset including manifest and all child scenario runs.

        Returns:
            Dict containing validation summary statistics.
        """
        if not self.dataset_dir.exists():
            raise DatasetValidationError(f"Dataset directory does not exist: {self.dataset_dir}")

        manifest_path = self.dataset_dir / "dataset_manifest.json"
        if not manifest_path.exists():
            raise DatasetValidationError(f"Missing dataset_manifest.json in {self.dataset_dir}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        required_manifest_keys = [
            "dataset_name",
            "version",
            "scenario_families",
            "total_scenarios",
            "total_runs",
            "total_frames",
            "scenarios",
        ]
        for k in required_manifest_keys:
            if k not in manifest:
                raise DatasetValidationError(f"Manifest missing required key: {k}")

        runs_checked = 0
        total_frames_checked = 0

        for sc_entry in manifest["scenarios"]:
            rel_dir = sc_entry["run_dir"]
            run_dir = self.dataset_dir / rel_dir
            if not run_dir.exists():
                raise DatasetValidationError(f"Run directory listed in manifest not found on disk: {run_dir}")

            run_stats = self.validate_run_directory(run_dir)
            runs_checked += 1
            total_frames_checked += run_stats["num_frames"]

        if runs_checked != manifest["total_runs"]:
            raise DatasetValidationError(
                f"Manifest total_runs ({manifest['total_runs']}) does not match validated runs ({runs_checked})"
            )
        if total_frames_checked != manifest["total_frames"]:
            raise DatasetValidationError(
                f"Manifest total_frames ({manifest['total_frames']}) does not match validated frames ({total_frames_checked})"
            )

        return {
            "status": "PASSED",
            "dataset_name": manifest["dataset_name"],
            "total_runs": runs_checked,
            "total_frames": total_frames_checked,
            "scenario_families": manifest["scenario_families"],
        }

    def validate_run_directory(self, run_dir: Path | str) -> dict[str, Any]:
        """Validate an individual run directory containing metadata and JSONL streams."""
        run_path = Path(run_dir)

        # 1. Check file existence
        meta_file = run_path / "metadata.json"
        gt_file = run_path / "ground_truth.jsonl"
        obs_file = run_path / "observations.jsonl"
        prism_file = run_path / "prism_output.jsonl"

        for f in (meta_file, gt_file, obs_file, prism_file):
            if not f.exists():
                raise DatasetValidationError(f"Required run file missing: {f}")

        # 2. Validate metadata
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        required_meta_fields = [
            "experiment_id",
            "scenario_id",
            "scenario_family",
            "random_seed",
            "scenario_version",
            "generator_version",
            "dt",
            "duration",
            "num_frames",
        ]
        for k in required_meta_fields:
            if k not in metadata:
                raise DatasetValidationError(f"Metadata in {run_path} missing key: {k}")

        dt = metadata["dt"]
        expected_frames = metadata["num_frames"]

        # 3. Read and validate line-by-line records
        gt_records = self._read_jsonl(gt_file)
        obs_records = self._read_jsonl(obs_file)
        prism_records = self._read_jsonl(prism_file)

        if len(gt_records) != expected_frames:
            raise DatasetValidationError(
                f"Ground truth line count ({len(gt_records)}) != metadata num_frames ({expected_frames})"
            )
        if len(obs_records) != expected_frames:
            raise DatasetValidationError(
                f"Observations line count ({len(obs_records)}) != metadata num_frames ({expected_frames})"
            )
        if len(prism_records) != expected_frames:
            raise DatasetValidationError(
                f"PRISM output line count ({len(prism_records)}) != metadata num_frames ({expected_frames})"
            )

        # 4. Rigorous frame-by-frame validation
        prev_time = -1e9
        gt_object_ids_seen: set[Any] = set()

        for idx, (gt, obs, pr) in enumerate(zip(gt_records, obs_records, prism_records)):
            t_gt = gt["timestamp"]
            t_obs = obs["timestamp"]
            t_pr = pr["timestamp"]

            # Strict time alignment across files
            if abs(t_gt - t_obs) > 1e-6 or abs(t_gt - t_pr) > 1e-6:
                raise DatasetValidationError(
                    f"Frame {idx} timestamp mismatch: GT={t_gt}, Obs={t_obs}, PRISM={t_pr}"
                )

            # Monotonicity check
            if t_gt <= prev_time:
                raise DatasetValidationError(
                    f"Frame {idx} non-monotonic timestamp: {t_gt} <= previous {prev_time}"
                )
            prev_time = t_gt

            # Frame index check
            if gt["frame_index"] != idx or obs["frame_index"] != idx or pr["frame_index"] != idx:
                raise DatasetValidationError(f"Frame index mismatch at step {idx}")

            # Ground Truth Validity Checks
            ego = gt["ego"]
            if not math.isfinite(ego["x"]) or not math.isfinite(ego["z"]):
                raise DatasetValidationError(f"Frame {idx} ego position has non-finite values: {ego}")

            for obj in gt["objects"]:
                obj_id = obj["object_id"]
                gt_object_ids_seen.add(obj_id)
                if not math.isfinite(obj["x"]) or not math.isfinite(obj["z"]):
                    raise DatasetValidationError(f"Frame {idx} object {obj_id} has non-finite coords: {obj}")

            # Collision Consistency Check
            col = gt["collision_state"]
            if col["has_collided"]:
                colliding_id = col["colliding_agent_id"]
                # Find the colliding agent
                coll_obj = next((o for o in gt["objects"] if o["object_id"] == colliding_id), None)
                if coll_obj is not None:
                    dx = coll_obj["x"] - ego["x"]
                    dz = coll_obj["z"] - ego["z"]
                    actual_dist = math.sqrt(dx * dx + dz * dz)
                    threshold = ego["radius"] + coll_obj["radius"] + 1e-3
                    if actual_dist > threshold:
                        raise DatasetValidationError(
                            f"Frame {idx} collision reported but physical distance ({actual_dist:.4f}) > radius sum ({threshold:.4f})"
                        )

            # Observation check (Zero future leakage: timestamp must match)
            for o in obs["observations"]:
                if o["valid"]:
                    if not math.isfinite(o["x"]) or not math.isfinite(o["z"]):
                        raise DatasetValidationError(f"Frame {idx} observation has non-finite coordinates: {o}")

            # PRISM Output Record check
            if pr["decision_action"] not in (
                "CONTINUE",
                "CAUTION",
                "SLOW_DOWN",
                "TURN_LEFT",
                "TURN_RIGHT",
                "STOP",
                "EMERGENCY_STOP",
            ):
                raise DatasetValidationError(f"Frame {idx} invalid PRISM action: {pr['decision_action']}")

        return {
            "status": "VALID",
            "scenario_id": metadata["scenario_id"],
            "num_frames": len(gt_records),
            "duration": metadata["duration"],
            "unique_gt_objects": len(gt_object_ids_seen),
        }

    def _read_jsonl(self, file_path: Path) -> list[dict[str, Any]]:
        """Read all lines from a JSONL file."""
        records: list[dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean = line.strip()
                if clean:
                    try:
                        records.append(json.loads(clean))
                    except json.JSONDecodeError as e:
                        raise DatasetValidationError(f"Invalid JSON in {file_path} at line {line_idx}: {e}")
        return records
