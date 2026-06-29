"""
MI-Motion pedestrian skeleton extraction and normalisation.

Dataset structure:
    data/MI-Motion/raw/
        S{scene_id}/
            S{scene_id}_{seq_id}.npy  ->  shape: (num_persons, num_frames, 20, 3)

    - S0~S3: 3 pedestrians per file
    - S4:    6 pedestrians per file
    - 215 sequence files in total, ~547,902 person-frame samples.

Coordinate system:
    - The Z axis is the height (up) direction.
    - Joint 0 (head) has the largest Z; joints 16/19 (feet) the smallest.
    - Raw coordinates are absolute world coordinates (units unknown,
      probably centimetres).


Output (in data/MI-Motion/):
    skeletons_raw.npy        : (N, 20, 3) float32, pelvis-centred, no scaling
    skeletons_normalized.npy : (N, 20, 3) float32, pelvis-centred + height-scaled
    metadata.npz             : per-sample provenance (scene/seq/person/frame ids)
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_DIR = Path("data/MI-Motion")
OUTPUT_DIR = Path("data/MI-Motion")

SCENES = ["S0", "S1", "S2", "S3", "S4"]

ROOT_JOINT = 12        # pelvis joint (translation centre)
HEAD_JOINT = 0         # head joint
FOOT_JOINTS = (16, 19) # left/right foot (mean Z used as the foot reference)

# Filter threshold: drop frames whose height is below this fraction of the median
MIN_HEIGHT_RATIO = 0.3


# ---------------------------------------------------------
# First pass: scan all frames to compute the median height
# (used to set the filtering threshold)
# ---------------------------------------------------------

def compute_height_median(data_dir: Path, scenes: list) -> float:
    """Compute the median head-to-foot height across all frames (raw coords)."""
    heights = []
    for scene in scenes:
        for npy_file in (data_dir / scene).glob("*.npy"):
            data = np.load(npy_file)
            n_persons, n_frames, _, _ = data.shape
            for pid in range(n_persons):
                skel = data[pid]                         # (n_frames, 20, 3)
                root_z = skel[:, ROOT_JOINT, 2]          # (n_frames,)
                head_z = skel[:, HEAD_JOINT, 2]
                foot_z = skel[:, list(FOOT_JOINTS), 2].mean(axis=1)
                h = (head_z - root_z) - (foot_z - root_z)  # = head_z - foot_z
                heights.append(h)
    heights = np.concatenate(heights)
    return float(np.median(heights))


# ---------------------------------------------------------
# Per-frame normalisation
# ---------------------------------------------------------

def normalize_skeleton(skeleton: np.ndarray, min_height: float):
    """
    Normalise a single skeleton frame.

    Args:
        skeleton:   (20, 3) raw joint coordinates.
        min_height: Minimum valid height; frames below this are discarded
                    (returns None).

    Returns:
        (raw, norm) tuple, or None for discarded frames.
            raw  (20, 3): pelvis-centred, no scaling
            norm (20, 3): pelvis-centred and height-scaled to ~1
    """
    root = skeleton[ROOT_JOINT]
    skel = skeleton - root          # pelvis at the origin

    head_z = skel[HEAD_JOINT, 2]
    foot_z = np.mean([skel[j, 2] for j in FOOT_JOINTS])
    height = head_z - foot_z

    if height < min_height:
        return None                 # discard anomalous frame

    norm = skel / height
    return skel.astype(np.float32), norm.astype(np.float32)


# ---------------------------------------------------------
# Main extraction pass
# ---------------------------------------------------------

def extract_and_normalize(data_dir: Path, scenes: list,
                          min_height: float) -> dict:
    """
    Iterate through every scene, extract and normalise each pedestrian frame.

    Returns:
        dict with:
          'skeletons_raw'        : (N, 20, 3) float32
          'skeletons_normalized' : (N, 20, 3) float32
          'scene_ids'            : (N,) int8
          'seq_ids'              : (N,) int16
          'person_ids'           : (N,) int8
          'frame_ids'            : (N,) int32
    """
    all_raw, all_norm = [], []
    all_scene, all_seq, all_person, all_frame = [], [], [], []
    total_processed = 0
    total_filtered = 0

    for scene in scenes:
        scene_id = int(scene[1:])
        files = sorted(
            (data_dir / scene).glob("*.npy"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        print(f"\n[Scene {scene}] {len(files)} sequences")

        for npy_file in tqdm(files, desc=f"  {scene}", leave=False):
            seq_id = int(npy_file.stem.split("_")[1])
            data = np.load(npy_file)
            n_persons, n_frames, n_joints, xyz = data.shape
            assert n_joints == 20 and xyz == 3

            for pid in range(n_persons):
                for fid in range(n_frames):
                    skel = data[pid, fid]       # (20, 3)
                    total_processed += 1

                    result = normalize_skeleton(skel, min_height)
                    if result is None:
                        total_filtered += 1
                        continue

                    raw, norm = result
                    all_raw.append(raw)
                    all_norm.append(norm)
                    all_scene.append(scene_id)
                    all_seq.append(seq_id)
                    all_person.append(pid)
                    all_frame.append(fid)

    print(f"\nTotal processed frames: {total_processed:,}")
    print(f"Discarded frames:        {total_filtered:,} "
          f"({total_filtered/total_processed*100:.2f}%)")
    print(f"Retained frames:         {total_processed - total_filtered:,}")

    return {
        "skeletons_raw":        np.array(all_raw,    dtype=np.float32),
        "skeletons_normalized": np.array(all_norm,   dtype=np.float32),
        "scene_ids":            np.array(all_scene,  dtype=np.int8),
        "seq_ids":              np.array(all_seq,    dtype=np.int16),
        "person_ids":           np.array(all_person, dtype=np.int8),
        "frame_ids":            np.array(all_frame,  dtype=np.int32),
    }



def print_stats(results: dict) -> None:
    skels_raw  = results["skeletons_raw"]
    skels_norm = results["skeletons_normalized"]
    N = skels_norm.shape[0]

    print(f"\n{'='*55}")
    print("Output dataset statistics")
    print(f"{'='*55}")
    print(f"Total samples (person-frames):  {N:,}")
    print(f"Per-sample shape:               (20, 3)")
    print(f"Dtype:                          {skels_norm.dtype}")
    print()

    for s in np.unique(results["scene_ids"]):
        mask = results["scene_ids"] == s
        print(f"  Scene S{s}: {mask.sum():,} frames")

    # Spatial extent of the pelvis-centred (un-scaled) skeletons
    print(f"\nPelvis-centred skeletons (skeletons_raw):")
    print(f"  X: [{skels_raw[...,0].min():.1f}, {skels_raw[...,0].max():.1f}]")
    print(f"  Y: [{skels_raw[...,1].min():.1f}, {skels_raw[...,1].max():.1f}]")
    print(f"  Z: [{skels_raw[...,2].min():.1f}, {skels_raw[...,2].max():.1f}]")

    # Spatial extent of the normalised skeletons
    print(f"\nNormalised skeletons (skeletons_normalized):")
    print(f"  X: [{skels_norm[...,0].min():.3f}, {skels_norm[...,0].max():.3f}]")
    print(f"  Y: [{skels_norm[...,1].min():.3f}, {skels_norm[...,1].max():.3f}]")
    print(f"  Z: [{skels_norm[...,2].min():.3f}, {skels_norm[...,2].max():.3f}]")

    # Verify that the height ~= 1
    head_z = skels_norm[:, HEAD_JOINT, 2]
    foot_z = skels_norm[:, list(FOOT_JOINTS), 2].mean(axis=1)
    heights = head_z - foot_z
    print(f"\n  Height (head_z - foot_z, should be ~1.0):")
    print(f"    mean:   {heights.mean():.4f}")
    print(f"    std:    {heights.std():.4f}")
    print(f"    range:  [{heights.min():.4f}, {heights.max():.4f}]")
    print(f"    p1/p99: {np.percentile(heights,1):.4f} / "
          f"{np.percentile(heights,99):.4f}")

    # Verify that the pelvis sits at the origin
    root_pos = skels_norm[:, ROOT_JOINT, :]
    print(f"\n  Pelvis joint (root, should be at origin):")
    print(f"    mean:           {root_pos.mean(axis=0)}")
    print(f"    max abs offset: {np.abs(root_pos).max():.2e}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("MI-Motion skeleton extraction")
    print("=" * 55)
    print(f"Data directory:   {DATA_DIR.resolve()}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    # Step 1: scan once to compute the median height (filtering threshold)
    print("\n[Step 1] Computing height distribution (for filtering)...")
    median_height = compute_height_median(DATA_DIR, SCENES)
    min_height = MIN_HEIGHT_RATIO * median_height
    print(f"  Median height:    {median_height:.2f} units")
    print(f"  Filter threshold: {min_height:.2f} units "
          f"(= {MIN_HEIGHT_RATIO:.0%} * median)")

    # Step 2: extract and normalise
    print("\n[Step 2] Extracting and normalising skeletons...")
    results = extract_and_normalize(DATA_DIR, SCENES, min_height)

    # Step 3: statistics / sanity checks
    print_stats(results)

    # Step 4: save
    print(f"\n[Step 4] Saving to {OUTPUT_DIR}/...")

    raw_path = OUTPUT_DIR / "skeletons_raw.npy"
    norm_path = OUTPUT_DIR / "skeletons_normalized.npy"
    meta_path = OUTPUT_DIR / "metadata.npz"

    np.save(raw_path, results["skeletons_raw"])
    print(f"  {raw_path.name:<35} {results['skeletons_raw'].nbytes / 1e6:.1f} MB")

    np.save(norm_path, results["skeletons_normalized"])
    print(f"  {norm_path.name:<35} {results['skeletons_normalized'].nbytes / 1e6:.1f} MB")

    np.savez(meta_path,
             scene_ids=results["scene_ids"],
             seq_ids=results["seq_ids"],
             person_ids=results["person_ids"],
             frame_ids=results["frame_ids"])
    print(f"  {meta_path.name:<35} (scene/seq/person/frame indices)")

    print("\nDone!")
    print("Usage:")
    print(f"  import numpy as np")
    print(f"  skels = np.load('{norm_path}')   # (N, 20, 3)")
    print(f"  meta  = np.load('{meta_path}')")


if __name__ == "__main__":
    main()
