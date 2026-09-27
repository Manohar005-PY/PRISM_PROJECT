"""Dataset loader for PRISM Part 20 Real-World Video Benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

from simulation.real_world.models import RealWorldRunMetadata


class RealWorldDatasetLoader:
    """Loads and resolves video runs and metadata from dataset_manifest.json."""

    def __init__(self, dataset_dir: Path | str = "experiments/datasets/real_world") -> None:
        self.dataset_dir: Path = Path(dataset_dir)
        self.manifest_path: Path = self.dataset_dir / "dataset_manifest.json"

    def load_manifest(self) -> dict[str, Any]:
        """Load raw dataset manifest dictionary."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def iter_runs(
        self,
        scenario_filter: str | None = None,
        run_filter: str | None = None,
    ) -> Iterator[RealWorldRunMetadata]:
        """Iterate over matching video runs from the dataset manifest."""
        manifest = self.load_manifest()
        runs_list = manifest.get("runs", [])

        for r_dict in runs_list:
            r_id = r_dict.get("run_id", "")
            sc_fam = r_dict.get("scenario_family", "")
            v_rel = r_dict.get("video_path", "")

            if scenario_filter and sc_fam != scenario_filter:
                continue
            if run_filter and r_id != run_filter:
                continue

            # Resolve video path on disk
            v_path = self.dataset_dir / v_rel
            if not v_path.exists():
                # Check alternative spelling (raw_vidoes vs raw_videos)
                alt_rel = v_rel.replace("raw_vidoes", "raw_videos") if "raw_vidoes" in v_rel else v_rel.replace("raw_videos", "raw_vidoes")
                v_path_alt = self.dataset_dir / alt_rel
                if v_path_alt.exists():
                    v_path = v_path_alt
                else:
                    continue  # Skip missing video

            yield RealWorldRunMetadata(
                run_id=r_id,
                scenario_family=sc_fam,
                video_path=str(v_path),
                has_collision=bool(r_dict.get("has_collision", False)),
                collision_timestamp_s=r_dict.get("collision_timestamp_s"),
                num_frames=int(r_dict.get("num_frames", 0)),
                duration_s=float(r_dict.get("duration_s", 0.0)),
                fps=float(r_dict.get("fps", 30.0)),
                resolution=tuple(r_dict.get("resolution", (720, 1280))),
                description=r_dict.get("description", ""),
                environment=r_dict.get("environment", "indoor"),
            )
