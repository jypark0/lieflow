import random
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.linalg import logm
from scipy.stats import beta as scipy_beta
from torch.distributions import Beta



def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fig_to_img(fig):
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]

    plt.close(fig)
    return img


def plot_2d_point_grid(x, titles=None):
    """Plot a grid of 2D point clouds as scatter plots.

    Generic helper used by the 2D arrow experiment to visualise a grid of
    learned samples.

    Args:
        x: [nr, nc, N, 2] array of 2D point clouds.
        titles: optional list of column titles (top row only).
    """
    nr, nc = x.shape[0], x.shape[1]
    fig = plt.figure(figsize=(1.5 * nc, 1.5 * nr), dpi=100)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(nr, nc),
        axes_pad=(0.05, 0.1),
        share_all=True,
    )

    for i in range(nr * nc):
        grid[i].set_aspect("equal")
        grid[i].invert_yaxis()
        grid[i].set(xlim=(-1, 1), ylim=(-1, 1), xticks=[], yticks=[])

        grid[i].scatter(
            x[i // nc, i % nc, :, 0], x[i // nc, i % nc, :, 1], color="k", s=1
        )

        if titles is not None and i // nc == 0:
            grid[i].set_title(titles[i])

    fig.canvas.draw()

    img_rgb = fig_to_img(fig)

    return img_rgb


def plot_histogram(angles):
    """Plot a histogram of angles (in degrees) over the range [-180, 180]."""
    fig, ax = plt.subplots(figsize=(12, 8))
    bins = np.linspace(-180, 180, 73) - 0.5  # every 5 degrees
    ax.hist(angles, bins=bins, color="k", alpha=0.7)
    ax.set(
        xlabel="Angle (degrees)",
        ylabel="Count",
        xticks=[-180, -135, -90, -45, 0, 45, 90, 135, 180],
        xlim=[-180, 180],
    )

    # Set larger font sizes
    ax.xaxis.label.set_fontsize(22)
    ax.yaxis.label.set_fontsize(22)
    ax.tick_params(axis="both", which="major", labelsize=20)

    fig.canvas.draw()

    img_rgb = fig_to_img(fig)

    return img_rgb


def blogm(mat):
    dtype = mat.dtype
    out = np.array([logm(m, disp=False)[0] for m in mat.cpu().numpy()])
    return torch.from_numpy(out).to(dtype=dtype, device=mat.device)




def expand_like(x, y):
    y_dim = y.dim()

    x = x.view(-1, *[1] * (y_dim - 1))

    return x.expand_as(y).contiguous()




def batched_object_div(v_func, x, t):
    orig_shape = x.shape
    x = x.detach().requires_grad_(True)

    x_flat = x.view(orig_shape[0], -1).requires_grad_(True)

    # Create wrapper that handles reshaping
    def v_func_flat(x_f, t_):
        x_mat = x_f.view(orig_shape)
        A = v_func(x_mat, t_)

        v = x_mat @ A.transpose(-2, -1)

        return v.view(orig_shape[0], -1)

    v_flat = v_func_flat(x_flat, t)

    div = torch.zeros(orig_shape[0], device=x.device, dtype=x.dtype)
    for i in range(x_flat.shape[1]):
        div += torch.autograd.grad(
            v_flat[:, i],
            x_flat,
            grad_outputs=torch.ones_like(v_flat[:, i]),
            retain_graph=True,
        )[0][:, i]

    return div


def sample_random_batch(dataset, batch_size, device, return_transform=False):
    """
    Sample a random batch from dataset.
    
    Supports two types of datasets:
    1. Datasets with .data and .transform attributes (e.g., DistributionSamplingDataset)
    2. Standard PyTorch datasets using __getitem__ (e.g., MIMotionSkeletonDataset)
    """
    rand_idx = torch.randint(len(dataset), (batch_size,))
    
    # Check if dataset has .data attribute (for DistributionSamplingDataset)
    if hasattr(dataset, 'data'):
        rand_batch = dataset.data[rand_idx]
        if return_transform:
            if hasattr(dataset, 'transform'):
                rand_transform = dataset.transform[rand_idx]
            else:
                # If dataset doesn't have transform, return identity transforms
                rand_transform = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    else:
        # Standard PyTorch dataset - use __getitem__
        batch_items = [dataset[int(idx)] for idx in rand_idx]
        
        # Check if items are tuples (data, transform)
        if isinstance(batch_items[0], tuple):
            # Dataset returns (data, transform)
            rand_batch = torch.stack([item[0] for item in batch_items])
            if return_transform:
                rand_transform = torch.stack([item[1] for item in batch_items])
        else:
            # Dataset returns only data
            rand_batch = torch.stack(batch_items)
            if return_transform:
                # Return identity transforms if no transform provided
                # For 3D point clouds, transform should be 3x3
                if rand_batch.shape[-1] == 3:
                    rand_transform = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
                else:
                    # Fallback for other dimensions
                    dim = rand_batch.shape[-1]
                    rand_transform = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Move to device
    if return_transform:
        if isinstance(rand_batch, torch.Tensor):
            rand_batch = rand_batch.to(device)
            rand_transform = rand_transform.to(device)
        else:
            rand_batch = torch.from_numpy(rand_batch).to(device)
            rand_transform = torch.from_numpy(rand_transform).to(device)
        return rand_batch, rand_transform
    else:
        if isinstance(rand_batch, torch.Tensor):
            rand_batch = rand_batch.to(device)
        else:
            rand_batch = torch.from_numpy(rand_batch).to(device)
        return rand_batch




def sample_power_distribution(
    a: float,
    b: float,
    skewness: float = 3.0,
    size: Union[int, Tuple[int, ...]] = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Sample from a continuous power distribution skewed towards upper bound b.

    Uses inverse transform method: X = a + (b - a) * U^(1/n)
    where U ~ Uniform(0, 1) and n is the skewness parameter.

    Args:
        a: Lower bound of distribution
        b: Upper bound of distribution
        skewness: Parameter controlling skew (higher = more skewed towards b)
        size: Shape of output tensor (int or tuple of ints)
        device: Device to place tensor on
        dtype: Data type of output tensor
        requires_grad: Whether to track gradients

    Returns:
        Tensor of samples from the skewed distribution

    Example:
        >>> samples = sample_power_distribution(0, 100, skewness=3, size=(1000,))
        >>> print(f"Mean: {samples.mean():.2f}")  # Should be around 75
    """
    if a >= b:
        raise ValueError(f"Lower bound {a} must be less than upper bound {b}")
    if skewness <= 0:
        raise ValueError(f"Skewness parameter must be positive, got {skewness}")

    # Convert size to tuple if it's an integer
    if isinstance(size, int):
        size = (size,)

    # Generate uniform random samples
    u = torch.rand(size, device=device, dtype=dtype, requires_grad=requires_grad)

    # Apply inverse transform
    samples = a + (b - a) * torch.pow(u, 1.0 / skewness)

    return samples


def icdf_power(
    a: float,
    b: float,
    n: int,
    skewness: float = 3.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create N evenly spaced values that follow the power distribution.

    Args:
        a: Lower bound
        b: Upper bound
        n: Number of points
        skewness: Parameter controlling skew
        device: Device to place tensors on
        dtype: Data type of output tensors

    """

    # Create evenly spaced quantiles (more values where probability is higher)
    quantiles = torch.linspace(0, 1, n, device=device, dtype=dtype)
    # Apply inverse CDF: X = a + (b - a) * q^(1/n)
    values = a + (b - a) * torch.pow(quantiles, 1.0 / skewness)

    return values


def sample_beta_distribution(alpha, beta, n, device="cpu", dtype=torch.float32):

    # Move parameters to device
    alpha = torch.tensor(alpha, device=device, dtype=dtype)
    beta = torch.tensor(beta, device=device, dtype=dtype)

    # Create distribution and sample
    dist = Beta(alpha, beta)
    samples = dist.sample((n,))

    return samples


def icdf_beta(
    alpha: float,
    beta: float,
    n: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # Create evenly spaced quantiles (more values where probability is higher)
    quantiles = np.linspace(0, 1, n)
    values = scipy_beta(alpha, beta).ppf(quantiles)

    values = torch.from_numpy(values).to(device=device, dtype=dtype)

    return values


EPS = 1e-5

def bracket_so3(w):
    """
    Bracket operation for so(3) Lie algebra.
    Converts between vector and matrix representations.
    """
    # vector -> matrix
    if w.shape[1:] == (3,):
        zeros = w.new_zeros(len(w))
        out = torch.stack([
            torch.stack([zeros, -w[:, 2], w[:, 1]], dim=1),
            torch.stack([w[:, 2], zeros, -w[:, 0]], dim=1),
            torch.stack([-w[:, 1], w[:, 0], zeros], dim=1)
        ], dim=1)
    # matrix -> vector
    elif w.shape[1:] == (3, 3):
        out = torch.stack([w[:, 2, 1], w[:, 0, 2], w[:, 1, 0]], dim=1)
    else:
        raise ValueError(f"bracket_so3: input must be of shape (N, 3) or (N, 3, 3). Current shape: {tuple(w.shape)}")
    return out






def log_SO3(R):
    """
    Logarithm map from SO(3) to so(3).
    Returns the matrix representation of the Lie algebra element.
    """
    n = R.shape[0]
    assert R.shape == (n, 3, 3), f"log_SO3: input must be of shape (N, 3, 3). Current shape: {tuple(R.shape)}"

    tr_R = torch.diagonal(R, dim1=1, dim2=2).sum(1)
    w_mat = torch.zeros_like(R)
    theta = torch.acos(torch.clamp((tr_R - 1) / 2, -1 + EPS, 1 - EPS))

    is_regular = (tr_R + 1 > EPS)
    is_singular = (tr_R + 1 <= EPS)

    theta = theta.unsqueeze(1).unsqueeze(2)

    if is_regular.any():
        w_mat_regular = (1 / (2 * torch.sin(theta[is_regular]) + EPS)) * (R[is_regular] - R[is_regular].transpose(1, 2)) * theta[is_regular]
        w_mat[is_regular] = w_mat_regular

    if is_singular.any():
        w_mat_singular = (R[is_singular] - torch.eye(3).to(R)) / 2
        w_vec_singular_raw = torch.sqrt(torch.diagonal(w_mat_singular, dim1=1, dim2=2) + 1)
        # Use torch.where to avoid inplace operation
        w_vec_singular = torch.where(torch.isnan(w_vec_singular_raw),
                                     torch.zeros_like(w_vec_singular_raw),
                                     w_vec_singular_raw)

        w_1 = w_vec_singular[:, 0]
        w_2 = w_vec_singular[:, 1] * (torch.sign(w_mat_singular[:, 0, 1]) + (w_1 == 0))
        w_3 = w_vec_singular[:, 2] * torch.sign(4 * torch.sign(w_mat_singular[:, 0, 2]) + 2 * (w_1 == 0) * torch.sign(w_mat_singular[:, 1, 2]) + 1 * (w_1 == 0) * (w_2 == 0))

        w_vec_singular = torch.stack([w_1, w_2, w_3], dim=1)
        w_mat[is_singular] = bracket_so3(w_vec_singular) * torch.pi

    return w_mat


def exp_so3(w_vec):
    """
    Exponential map from so(3) to SO(3).
    Input can be either vector (N, 3) or matrix (N, 3, 3) representation.
    """
    if w_vec.shape[1:] == (3, 3):
        w_vec = bracket_so3(w_vec)
    elif w_vec.shape[1:] != (3,):
        raise ValueError(f"exp_so3: input must be of shape (N, 3) or (N, 3, 3). Current shape: {tuple(w_vec.shape)}")

    # Create identity matrices (not connected to w_vec's grad graph)
    R = torch.eye(3, device=w_vec.device, dtype=w_vec.dtype).unsqueeze(0).repeat(len(w_vec), 1, 1)
    theta = w_vec.norm(dim=1)

    is_regular = theta > EPS
    is_singular = theta <= EPS


    if is_regular.any():
        w_vec_regular = w_vec[is_regular]
        theta_regular = theta[is_regular].unsqueeze(1)
        w_mat_hat_regular = bracket_so3(w_vec_regular / theta_regular)
        theta_regular = theta_regular.unsqueeze(2)
        R_regular = torch.eye(3, device=w_vec.device, dtype=w_vec.dtype).unsqueeze(0).repeat(len(w_vec_regular), 1, 1) + \
                    torch.sin(theta_regular) * w_mat_hat_regular + \
                    (1 - torch.cos(theta_regular)) * w_mat_hat_regular @ w_mat_hat_regular

        # Avoid inplace: create new tensor
        R = R.clone()
        R[is_regular] = R_regular

    # Orthogonalize to ensure R is a valid rotation matrix (differentiable)
    # Use SVD: R = U @ S @ V^T, nearest rotation is U @ V^T
    U, _, Vt = torch.linalg.svd(R)
    # Ensure proper rotation (det = +1), not reflection (det = -1)
    det_sign = torch.det(U @ Vt).unsqueeze(-1).unsqueeze(-1)
    # Create diagonal correction matrix
    S = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(R.shape[0], -1, -1).clone()
    S[:, 2, 2] = det_sign.squeeze(-1).squeeze(-1)
    R = U @ S @ Vt

    return R










