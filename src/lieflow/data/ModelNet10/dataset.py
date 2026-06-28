"""
ModelNet10 dataset loader.
Each class has configurable number of samples, each sample is uniformly sampled to n_points.
Can be used with ObjectTransformDistribution for flexible group augmentation.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torch_geometric.datasets import ModelNet


class ModelNet10Loader:
    """
    Simple ModelNet10 loader that loads point clouds without augmentation.
    Similar to Object class - provides base point cloud data.
    Can be used with ObjectTransformDistribution for flexible group augmentation.
    
    Args:
        root: Root directory of ModelNet10 dataset
        split: 'train' or 'test'
        n_points: Number of points to sample per shape (default: 128)
        n_samples_per_class: Number of samples per class (default: 10)
        total_samples: Total number of samples to load (overrides n_samples_per_class if set)
        random_seed: Random seed for reproducible sampling (default: 42)
    """
    
    def __init__(
        self,
        root,
        split="train",
        n_points=128,
        n_samples_per_class=10,
        total_samples=None,
        random_seed=42,
    ):
        self.root = root
        self.split = split
        self.n_points = n_points
        self.n_samples_per_class = n_samples_per_class
        self.total_samples = total_samples
        self.random_seed = random_seed
        
        # Load ModelNet10 data (without augmentation)
        self.data = self._load_modelnet10()
        self.name = f"modelnet10_{split}"
        
    def _load_modelnet10(self):
        """
        Load ModelNet10 dataset and return all point clouds as a numpy array.

        Downloads via torch_geometric if missing, then loads from local .off
        files. If torch_geometric has already processed the dataset to
        ``processed/training.pt``, uses that cache directly.

        Returns:
            data: numpy array of shape [num_samples, n_points, 3]
        """
        root_path = Path(self.root)
        processed_file = root_path / "processed" / (
            "training.pt" if self.split == "train" else "test.pt"
        )

        if processed_file.exists():
            dataset = ModelNet(self.root, name="10", train=(self.split == "train"))
            return self._load_from_torch_geometric(dataset)

        self._ensure_modelnet10_downloaded(root_path)
        return self._load_from_local_files(root_path)

    def _ensure_modelnet10_downloaded(self, root_path: Path) -> None:
        """Download ModelNet10 using torch_geometric if raw files are missing."""
        raw_path = root_path / "raw"
        if raw_path.exists() and any(d.is_dir() for d in raw_path.iterdir()):
            return

        import os

        from torch_geometric.data import download_url, extract_zip
        from torch_geometric.io import fs

        root_path.mkdir(parents=True, exist_ok=True)
        print(f"Downloading ModelNet10 via torch_geometric to {root_path} ...")
        zip_path = download_url(ModelNet.urls["10"], str(root_path))
        extract_zip(zip_path, str(root_path))
        os.unlink(zip_path)

        extracted = root_path / "ModelNet10"
        if raw_path.exists():
            fs.rm(str(raw_path))
        extracted.rename(raw_path)

        macosx_dir = root_path / "__MACOSX"
        if macosx_dir.exists():
            fs.rm(str(macosx_dir))
        print(f"ModelNet10 ready at {raw_path}")
    
    def _load_from_torch_geometric(self, dataset):
        """Load data from torch_geometric ModelNet dataset."""
        # Use a fixed random state for reproducibility
        rng = np.random.RandomState(self.random_seed)
        
        # If total_samples is specified, randomly select that many samples total
        if self.total_samples is not None:
            all_indices = list(range(len(dataset)))
            selected_indices = rng.choice(
                all_indices,
                size=min(self.total_samples, len(all_indices)),
                replace=False
            )
            
            selected_data = []
            for idx in selected_indices:
                point_cloud = dataset[idx].pos.numpy()  # [N, 3]
                point_cloud = self._sample_points(point_cloud, idx)
                selected_data.append(point_cloud)
            
            return np.array(selected_data, dtype=np.float32)
        
        # Otherwise, use n_samples_per_class strategy
        # Group samples by class
        class_to_indices = {}
        for idx, data in enumerate(dataset):
            label = int(data.y.item())
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        # Select n_samples_per_class from each class
        selected_data = []
        
        for label, indices in class_to_indices.items():
            # Randomly select n_samples_per_class samples
            selected_indices = rng.choice(
                indices, 
                size=min(self.n_samples_per_class, len(indices)), 
                replace=False
            )
            
            for idx in selected_indices:
                point_cloud = dataset[idx].pos.numpy()  # [N, 3]
                point_cloud = self._sample_points(point_cloud, idx)
                selected_data.append(point_cloud)
        
        return np.array(selected_data, dtype=np.float32)
    
    def _load_from_local_files(self, root_path):
        """Load ModelNet10 from local file system.

        Supports three directory structures:
        1. Standard: root/train/class1/sample1.off
        2. torch_geometric raw: root/raw/class1/train/sample1.off
        3. Legacy: root/modelnet/raw/class1/train/sample1.off
        """
        split_dir = root_path / ("train" if self.split == "train" else "test")

        if split_dir.exists():
            return self._load_from_standard_structure(split_dir)

        pyg_raw = root_path / "raw"
        if pyg_raw.exists():
            return self._load_from_alternative_structure(pyg_raw)

        alt_path = root_path / "modelnet" / "raw"
        if alt_path.exists():
            return self._load_from_alternative_structure(alt_path)

        raise FileNotFoundError(
            f"ModelNet10 dataset not found under {root_path}. "
            "Expected root/train, root/raw, or root/modelnet/raw."
        )

    def _load_from_standard_structure(self, split_dir):
        """Load from root/train/class1/sample1.off."""
        selected_data = []
        rng = np.random.RandomState(self.random_seed)
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        if self.total_samples is not None:
            all_off_files = []
            for class_dir in class_dirs:
                all_off_files.extend(list(class_dir.glob("*.off")))

            selected_files = rng.choice(
                all_off_files,
                size=min(self.total_samples, len(all_off_files)),
                replace=False,
            )

            for file_idx, off_file in enumerate(selected_files):
                try:
                    point_cloud = self._load_off_file(off_file)
                    point_cloud = self._sample_points(point_cloud, file_idx)
                    selected_data.append(point_cloud)
                except Exception as e:
                    print(f"Warning: Failed to load {off_file}: {e}")
        else:
            for class_dir in class_dirs:
                off_files = sorted(list(class_dir.glob("*.off")))
                if len(off_files) == 0:
                    continue

                selected_files = rng.choice(
                    off_files,
                    size=min(self.n_samples_per_class, len(off_files)),
                    replace=False,
                )

                for file_idx, off_file in enumerate(selected_files):
                    try:
                        point_cloud = self._load_off_file(off_file)
                        point_cloud = self._sample_points(point_cloud, file_idx)
                        selected_data.append(point_cloud)
                    except Exception as e:
                        print(f"Warning: Failed to load {off_file}: {e}")

        return np.array(selected_data, dtype=np.float32)
    
    def _load_from_alternative_structure(self, raw_path):
        """Load from alternative structure: modelnet/raw/class1/train/sample1.off"""
        selected_data = []
        rng = np.random.RandomState(self.random_seed)
        
        # Get all class directories
        class_dirs = sorted([d for d in raw_path.iterdir() if d.is_dir()])
        
        split_name = "train" if self.split == "train" else "test"
        
        # If total_samples is specified, collect all files first then sample
        if self.total_samples is not None:
            all_off_files = []
            for class_dir in class_dirs:
                split_dir = class_dir / split_name
                if split_dir.exists():
                    all_off_files.extend(list(split_dir.glob("*.off")))
            
            # Randomly select total_samples files
            selected_files = rng.choice(
                all_off_files,
                size=min(self.total_samples, len(all_off_files)),
                replace=False
            )
            
            for file_idx, off_file in enumerate(selected_files):
                try:
                    point_cloud = self._load_off_file(off_file)
                    point_cloud = self._sample_points(point_cloud, file_idx)
                    selected_data.append(point_cloud)
                except Exception as e:
                    print(f"Warning: Failed to load {off_file}: {e}")
                    continue
        else:
            # Use n_samples_per_class strategy
            for class_idx, class_dir in enumerate(class_dirs):
                # Get split directory (train or test) inside class directory
                split_dir = class_dir / split_name
                if not split_dir.exists():
                    continue
                
                # Get all .off files in this split
                off_files = sorted(list(split_dir.glob("*.off")))
                
                if len(off_files) == 0:
                    continue
                
                # Select n_samples_per_class files
                selected_files = rng.choice(
                    off_files,
                    size=min(self.n_samples_per_class, len(off_files)),
                    replace=False
                )
                
                for file_idx, off_file in enumerate(selected_files):
                    try:
                        point_cloud = self._load_off_file(off_file)
                        point_cloud = self._sample_points(point_cloud, file_idx)
                        selected_data.append(point_cloud)
                    except Exception as e:
                        print(f"Warning: Failed to load {off_file}: {e}")
                        continue
        
        return np.array(selected_data, dtype=np.float32)
    
    def _load_off_file(self, file_path):
        """Load point cloud from .off file."""
        try:
            # Try using trimesh first
            import trimesh
            mesh = trimesh.load(str(file_path))
            if hasattr(mesh, 'vertices'):
                return mesh.vertices.astype(np.float32)
            else:
                # If it's a scene, get vertices from first mesh
                if hasattr(mesh, 'geometry'):
                    return list(mesh.geometry.values())[0].vertices.astype(np.float32)
        except ImportError:
            # Fallback: parse .off file manually
            pass
        except Exception:
            # If trimesh fails, try manual parsing
            pass
        
        # Manual .off file parsing
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header
        idx = 0
        while idx < len(lines) and lines[idx].strip().startswith('#'):
            idx += 1
        
        # Read header: two common formats:
        #   (a) "OFF 3514 3546 0"  — all on one line
        #   (b) "OFF\n3514 3546 0" — OFF alone, counts on next line
        header = lines[idx].strip().split()
        if header[0].upper() != 'OFF':
            # No "OFF" prefix — first token is already the vertex count
            num_verts = int(header[0])
            idx += 1
        elif len(header) >= 2:
            # Format (a): "OFF num_verts num_faces ..."
            num_verts = int(header[1])
            idx += 1
        else:
            # Format (b): "OFF" alone — counts are on the next line
            idx += 1
            size_header = lines[idx].strip().split()
            num_verts = int(size_header[0])
            idx += 1
        
        # Read vertices
        vertices = []
        for i in range(num_verts):
            if idx >= len(lines):
                break
            coords = lines[idx].strip().split()
            if len(coords) >= 3:
                vertices.append([float(coords[0]), float(coords[1]), float(coords[2])])
            idx += 1
        
        return np.array(vertices, dtype=np.float32)
    
    def _sample_points(self, point_cloud, idx):
        """Sample n_points from point cloud."""
        if len(point_cloud) > self.n_points:
            # Random sampling with fixed seed per sample
            sample_rng = np.random.RandomState(self.random_seed + idx)
            sample_indices = sample_rng.choice(
                len(point_cloud), 
                size=self.n_points, 
                replace=False
            )
            point_cloud = point_cloud[sample_indices]
        elif len(point_cloud) < self.n_points:
            # Repeat points if needed
            n_repeats = (self.n_points // len(point_cloud)) + 1
            point_cloud = np.tile(point_cloud, (n_repeats, 1))[:self.n_points]
        return point_cloud
