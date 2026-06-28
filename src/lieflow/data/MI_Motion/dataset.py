"""
MI-Motion Skeleton Dataset

Loads pre-extracted, normalized pedestrian skeleton frames produced by
extract_mi_motion_skeletons.py. Each sample is a single-frame skeleton
represented as a (20, 3) point cloud in the pelvis-centered, height-
normalized coordinate system.

Data file:
    data/MI-Motion/skeletons_normalized.npy  →  (N, 20, 3) float32

Compatible with the Flow_SO3 training loop in
    src/lieflow/flow_matching/arrow/model.py

The dataset returns raw point clouds (no transform); random SO3 rotations
are applied inside Flow_SO3.train_net / eval_net.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MIMotionSkeletonDataset(Dataset):
    """
    PyTorch Dataset for MI-Motion normalized skeleton frames.

    Args:
        data_path:    Path to ``skeletons_normalized.npy`` (shape N×20×3).
        split:        ``'train'`` or ``'test'``.
        train_ratio:  Fraction of samples used for training (default 0.9).
        num_samples:  If set, subsample this many frames from the split.
                      Useful to keep epoch length manageable (e.g. 100_000).
        random_seed:  Seed for reproducible train/test split and subsampling.
        metadata_path: Optional path to ``metadata.npz``; loaded but not
                       returned by ``__getitem__`` (available as
                       ``dataset.metadata`` for inspection).

    Attributes:
        data:     FloatTensor of shape (N_split, 20, 3).
        metadata: dict with keys ``scene_ids``, ``seq_ids``, ``person_ids``,
                  ``frame_ids`` (LongTensors, same length as ``data``).
                  None if ``metadata_path`` is not provided.
    """

    N_JOINTS = 20
    JOINT_DIM = 3

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        train_ratio: float = 0.9,
        num_samples: Optional[int] = None,
        random_seed: int = 42,
        metadata_path: Optional[str] = None,
    ):
        assert split in ("train", "test"), \
            f"split must be 'train' or 'test', got '{split}'"
        assert 0.0 < train_ratio < 1.0

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Skeleton data not found: {data_path}\n"
                "Run extract_mi_motion_skeletons.py first to generate it."
            )

        # Load full array  (N_total, 20, 3)
        raw = np.load(data_path)
        assert raw.ndim == 3 and raw.shape[1:] == (self.N_JOINTS, self.JOINT_DIM), (
            f"Expected shape (N, {self.N_JOINTS}, {self.JOINT_DIM}), got {raw.shape}"
        )

        N_total = raw.shape[0]

        # Deterministic train/test split by index
        rng = np.random.default_rng(random_seed)
        perm = rng.permutation(N_total)

        n_train = int(N_total * train_ratio)
        if split == "train":
            indices = perm[:n_train]
        else:
            indices = perm[n_train:]

        # Optional subsampling (deterministic, same seed)
        if num_samples is not None and num_samples < len(indices):
            indices = rng.choice(indices, size=num_samples, replace=False)

        # Store as float32 tensor on CPU; DataLoader moves to device
        self.data = torch.from_numpy(raw[indices].astype(np.float32))  # (N, 20, 3)

        # Metadata (optional)
        self.metadata = None
        if metadata_path is not None:
            meta_path = Path(metadata_path)
            if meta_path.exists():
                meta = np.load(meta_path)
                self.metadata = {
                    k: torch.from_numpy(meta[k][indices].astype(np.int64))
                    for k in ("scene_ids", "seq_ids", "person_ids", "frame_ids")
                }

        self.split = split
        self.name = f"mi_motion_skeleton_{split}"

    # ─────────────────────────────────────────────────────
    # Dataset protocol
    # ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single skeleton as FloatTensor of shape (20, 3)."""
        return self.data[idx]

    # ─────────────────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MIMotionSkeletonDataset("
            f"split={self.split}, "
            f"n_samples={len(self)}, "
            f"shape={tuple(self.data.shape[1:])})"
        )
