"""
Interleaved multi-angle dataset for MASt3R-SLAM.

Given a parent directory containing multiple angle subfolders (e.g. cam0_p+0_y+0_r+0,
cam0_p+0_y+30_r+0, ...), this module builds an interleaved sequence so that each
timestamp cycles through all angles before advancing to the next timestamp.

The angle order is chosen to minimise the maximum angular gap between consecutive
frames (greedy nearest-neighbour tour), which keeps visual overlap high for tracking.

Usage from CLI:
    python SCRIPT_MAIN_Pipeline.py \
        --dataset "E:/.../staged_images" \
        --angles "p+0_y+0_r+0,p+0_y+30_r+0,p+30_y+0_r+0,p-30_y+0_r+0" \
        --cam cam0 \
        --rerun
"""

from __future__ import annotations

import itertools
import math
import pathlib
import re
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from natsort import natsorted

from mast3r_slam.dataloader import MonocularDataset


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _parse_angle(angle_str: str) -> Tuple[float, float, float]:
    """Extract (pitch, yaw, roll) in degrees from a string like 'p+30_y+0_r+0'."""
    m = re.match(
        r"p([+-]?\d+(?:\.\d+)?)_y([+-]?\d+(?:\.\d+)?)_r([+-]?\d+(?:\.\d+)?)", angle_str
    )
    if not m:
        raise ValueError(f"Cannot parse angle string: {angle_str!r}")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def _angular_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance in angle-space (degrees)."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _optimal_angle_order(angles: List[str]) -> List[str]:
    """Return angle list reordered to minimise the max consecutive angular gap.

    With only 4 angles we can brute-force all permutations (24 total) and pick the
    one whose largest step (including wrap-around back to the first angle for the
    next timestamp) is smallest.
    """
    if len(angles) <= 2:
        return angles

    parsed = {a: _parse_angle(a) for a in angles}

    best_order = None
    best_max_gap = float("inf")

    for perm in itertools.permutations(angles):
        gaps = []
        for i in range(len(perm)):
            a = parsed[perm[i]]
            b = parsed[perm[(i + 1) % len(perm)]]
            gaps.append(_angular_distance(a, b))
        mg = max(gaps)
        if mg < best_max_gap:
            best_max_gap = mg
            best_order = list(perm)

    return best_order


def _image_key(filename: str) -> str:
    """Extract the timestamp key shared across angle folders.

    Filename pattern: ``000000_72966704786293_72966704786293_cam0_p+0_y+0_r+0.jpg``
    Key = first three underscore-separated tokens (frame_idx + two timestamp fields).
    """
    stem = pathlib.Path(filename).stem
    parts = stem.split("_")
    return "_".join(parts[:3])


def _extract_timestamp(key: str) -> float:
    """Convert a key like '000000_72966704786293_72966704786293' to seconds.

    Uses the second field (capture timestamp).  The value looks like nanoseconds
    or a high-precision clock; we divide by 1e9 to get seconds.
    """
    parts = key.split("_")
    return float(parts[1]) / 1e9


class InterleavedRGBFiles(MonocularDataset):
    """Dataset that interleaves images from multiple angle subfolders.

    Parameters
    ----------
    parent_dir : str or Path
        Directory that *contains* the angle subfolders.
    angles : list[str]
        Angle suffixes, e.g. ``["p+0_y+0_r+0", "p+0_y+30_r+0", ...]``.
    cam : str
        Camera prefix, e.g. ``"cam0"``.  Subfolders are expected to be named
        ``{cam}_{angle}`` (e.g. ``cam0_p+0_y+30_r+0``).
    optimize_order : bool
        If True (default), reorder angles to minimise max angular gap.
    """

    def __init__(
        self,
        parent_dir: str | pathlib.Path,
        angles: Sequence[str],
        cam: str = "cam0",
        optimize_order: bool = True,
    ):
        super().__init__()
        self.use_calibration = False
        self.parent_dir = pathlib.Path(parent_dir)
        self.dataset_path = self.parent_dir
        self.cam = cam

        if optimize_order:
            self.angles = _optimal_angle_order(list(angles))
        else:
            self.angles = list(angles)

        folder_paths = []
        for angle in self.angles:
            p = self.parent_dir / f"{cam}_{angle}"
            if not p.is_dir():
                raise FileNotFoundError(f"Angle folder not found: {p}")
            folder_paths.append(p)

        files_per_angle: list[dict[str, str]] = []
        for folder in folder_paths:
            files: list[str] = []
            for ext in IMG_EXTENSIONS:
                files.extend(str(f) for f in folder.glob(f"*{ext}"))
            files = natsorted(files)
            key_map = {_image_key(f): f for f in files}
            files_per_angle.append(key_map)

        all_keys = set(files_per_angle[0].keys())
        for km in files_per_angle[1:]:
            all_keys &= km.keys()
        if not all_keys:
            raise FileNotFoundError(
                f"No common timestamps found across angle folders in {self.parent_dir}"
            )
        sorted_keys = natsorted(all_keys)

        self.rgb_files = []
        self.angle_indices = []
        self.frame_keys = []
        self.timestamps = []
        n_angles = len(self.angles)
        for key in sorted_keys:
            base_ts = _extract_timestamp(key)
            for ai, km in enumerate(files_per_angle):
                self.rgb_files.append(km[key])
                self.angle_indices.append(ai)
                self.frame_keys.append(key)
                # Co-temporal frames share the same base timestamp;
                # add a tiny per-angle offset (microseconds) for uniqueness.
                self.timestamps.append(base_ts + ai * 1e-6)
        self.timestamps = np.array(self.timestamps, dtype=self.dtype)

        self.n_angles = n_angles
        self.n_timestamps = len(sorted_keys)

    def summary(self) -> str:
        lines = [
            f"  InterleavedRGBFiles",
            f"  Parent:       {self.parent_dir}",
            f"  Camera:       {self.cam}",
            f"  Angles ({self.n_angles}):  {self.angles}",
            f"  Timestamps:   {self.n_timestamps}",
            f"  Total frames: {len(self.rgb_files)} ({self.n_timestamps} × {self.n_angles})",
        ]
        angle_pyr = [_parse_angle(a) for a in self.angles]
        gaps = []
        for i in range(len(angle_pyr)):
            d = _angular_distance(angle_pyr[i], angle_pyr[(i + 1) % len(angle_pyr)])
            gaps.append(d)
        lines.append(f"  Angle order gaps: {['%.1f°' % g for g in gaps]}  (max {max(gaps):.1f}°)")
        return "\n".join(lines)
