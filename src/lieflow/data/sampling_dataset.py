from torch.utils.data import Dataset
import torch
import math


class DistributionSamplingDataset(Dataset):
    def __init__(self, dist, num_samples, return_transform=False, flatten_to_vector=False):
        self.dist = dist
        self.num_samples = num_samples
        self.return_transform = return_transform
        self.flatten_to_vector = flatten_to_vector

        if return_transform:
            self.data, self.transform = self.dist.sample(
                (self.num_samples,), return_transform=True
            )

        else:
            self.data = self.dist.sample((self.num_samples,))
        
        # Flatten data if needed (e.g., for triangle: [B, 1, 6] -> [B, 6])
        if self.flatten_to_vector and len(self.data.shape) == 3 and self.data.shape[1] == 1:
            self.data = self.data.squeeze(1)  # [B, 1, D] -> [B, D]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Ensure idx is a single integer, not a slice or array
        if isinstance(idx, (list, tuple, slice)):
            raise ValueError(f"__getitem__ received {type(idx)} instead of int. "
                           f"This may indicate a DataLoader configuration issue.")
        
        # Convert to int if it's a tensor
        if hasattr(idx, 'item'):
            idx = idx.item()
        
        idx = int(idx)
        
        if self.return_transform:
            # Return a copy to ensure DataLoader can properly batch
            item = self.data[idx]
            transform = self.transform[idx]
            
            # Ensure we're indexing the first dimension correctly
            # If data is [B, N, D], then data[idx] should be [N, D]
            # If data is [B, D], then data[idx] should be [D]
            # Flatten any extra dimensions
            while len(item.shape) > 2:
                # If item has shape [..., N, D], flatten leading dimensions
                item = item.reshape(-1, item.shape[-1])
            # If item is [N, D] with N=1, squeeze to [D]
            if len(item.shape) == 2 and item.shape[0] == 1:
                item = item.squeeze(0)  # [1, D] -> [D]
            
            return item.clone(), transform.clone()
        else:
            # Return a copy to ensure DataLoader can properly batch
            item = self.data[idx]
            
            # Ensure we're indexing the first dimension correctly
            # Flatten any extra dimensions
            while len(item.shape) > 2:
                item = item.reshape(-1, item.shape[-1])
            # If item is [N, D] with N=1, squeeze to [D]
            if len(item.shape) == 2 and item.shape[0] == 1:
                item = item.squeeze(0)  # [1, D] -> [D]
            
            return item.clone()






class MultiSymmetryDataset(Dataset):
    """
    Dataset that mixes multiple objects each with a **different** symmetry group.

    Mixes several objects (one per symmetry group) and also stores a per-sample
    category label so that evaluation code can assess each symmetry type separately.

    Each element of ``dists`` must be an ``ObjectTransformDistribution`` (one
    fixed object + one symmetry distribution).  The label stored for samples
    drawn from ``dists[i]`` is ``i``.

    The ``dists`` attribute is preserved so callers can inspect
    ``dists[i].base_dist`` to determine the ground-truth rotations for category i.

    Args:
        dists: list of ObjectTransformDistribution objects (one per symmetry type)
        num_samples: total number of samples to pre-generate
        weights: sampling weights per distribution (default: uniform)
        return_transform: whether to pre-compute and store transformation matrices
    """

    def __init__(self, dists, num_samples, weights=None, return_transform=False):
        self.dists = dists
        self.num_samples = num_samples
        self.return_transform = return_transform

        n_dists = len(dists)
        if weights is None:
            weights = [1.0 / n_dists] * n_dists
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights]

        # Compute per-distribution sample counts
        sample_counts = []
        remaining = num_samples
        for w in weights[:-1]:
            n = int(num_samples * w)
            sample_counts.append(n)
            remaining -= n
        sample_counts.append(remaining)

        # Sample from all distributions
        all_data = []
        all_transforms = []
        all_labels = []

        for cat_idx, (dist, n) in enumerate(zip(dists, sample_counts)):
            if return_transform:
                data, transforms = dist.sample((n,), return_transform=True)
                all_transforms.append(transforms)
            else:
                data = dist.sample((n,))
            all_data.append(data)
            all_labels.append(torch.full((n,), cat_idx, dtype=torch.long))

        # Pad all data to the same number of vertices (in case objects differ)
        max_verts = max(
            d.shape[1] if len(d.shape) >= 2 else 1 for d in all_data
        )
        padded_data = []
        for data in all_data:
            n_verts = data.shape[1] if len(data.shape) >= 2 else 1
            if n_verts < max_verts and len(data.shape) == 3:
                padding_size = max_verts - n_verts
                last_vertex = data[:, -1:, :]
                padding = last_vertex.repeat(1, padding_size, 1)
                data = torch.cat([data, padding], dim=1)
            padded_data.append(data)

        self.data = torch.cat(padded_data, dim=0)
        self.labels = torch.cat(all_labels, dim=0)
        if return_transform:
            self.transform = torch.cat(all_transforms, dim=0)
        else:
            self.transform = None

        # Shuffle
        indices = torch.randperm(self.data.shape[0])
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        if return_transform:
            self.transform = self.transform[indices]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, slice)):
            raise ValueError(
                f"__getitem__ received {type(idx)} instead of int."
            )
        if hasattr(idx, "item"):
            idx = idx.item()
        idx = int(idx)

        item = self.data[idx]
        while len(item.shape) > 2:
            item = item.reshape(-1, item.shape[-1])
        if len(item.shape) == 2 and item.shape[0] == 1:
            item = item.squeeze(0)

        if self.return_transform:
            transform = self.transform[idx]
            return item.clone(), transform.clone()
        else:
            return item.clone()


