from pathlib import Path

import hydra
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from hydra.utils import instantiate
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import ImageGrid
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import trange

from lieflow.geometry_2d import matrix_to_angle, polar_decomposition
from lieflow.logger import get_logger
from lieflow.utils import (
    fig_to_img,
    plot_2d_point_grid,
    plot_histogram,
    sample_random_batch,
    seed_all,
)

expm = torch.linalg.matrix_exp


def compute_wasserstein_distance(
        learned_transforms, test_tf
    ):
    """
    Compute Wasserstein-1 distance between learned group elements and test_tf distribution.
    
    Treats each 2x2 matrix as a 4D vector in R^4 and computes the true Wasserstein-1
    distance using optimal transport (POT library).
    
    Args:
        learned_transforms: [B, 2, 2] learned transformation matrices
        test_tf: [B, 2, 3] test transformation matrices
        
    Returns:
        wasserstein_dist: Wasserstein-1 distance between matrix distributions
    """

        


    # Handle complex matrices by converting to real representation
    if learned_transforms.is_complex():
        # For complex matrices, use only the real part for Wasserstein distance
        learned_matrices = learned_transforms.cpu().numpy()  # [B, 2, 2] complex
        learned_vectors_real = learned_matrices.real.reshape(learned_matrices.shape[0], -1)  # [B, 4]
        learned_vectors_imag = learned_matrices.imag.reshape(learned_matrices.shape[0], -1)  # [B, 4]
        learned_vectors = np.concatenate([learned_vectors_real, learned_vectors_imag], axis=-1)  # [B, 8]
    else:
        learned_matrices = learned_transforms.cpu().numpy()  # [B, 2, 2]
        learned_vectors = learned_matrices.reshape(learned_matrices.shape[0], -1)  # [B, 4]
    
    n_learned = learned_vectors.shape[0]

    # Handle test samples similarly
    if test_tf.is_complex():
        test_samples = test_tf.cpu().numpy()  # [B, 2, 2] complex
        test_vectors_real = test_samples.real.reshape(test_samples.shape[0], -1)  # [B, 4]
        test_vectors_imag = test_samples.imag.reshape(test_samples.shape[0], -1)  # [B, 4]
        test_vectors = np.concatenate([test_vectors_real, test_vectors_imag], axis=-1)  # [B, 8]
    else:
        test_samples = test_tf.cpu().numpy()  # [B, 2, 2]
        test_vectors = test_samples.reshape(test_samples.shape[0], -1)  # [B, 4]

    M = ot.dist(learned_vectors, test_vectors, metric='euclidean')  # [n_learned, n_test]
    

    a = np.ones(n_learned) / n_learned  # uniform distribution over learned samples

    b = np.ones(test_vectors.shape[0]) / test_vectors.shape[0]  # uniform distribution over test samples

    wasserstein_dist = ot.emd2(a, b, M)  # Earth Mover's Distance
    
    return wasserstein_dist


def plot_progression_grid(x_t, tf_t, x_1, titles=None):
    n_matrices = 5
    nr, nc = n_matrices, x_t.shape[0]  # 5 rows, T columns
    fig = plt.figure(figsize=(2.0 * nc + 2, 2.0 * nr), dpi=100)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(nr, nc),
        axes_pad=(0.0, 0.0),
        share_all=False,
    )

    x_t = x_t.cpu()
    tf_t = tf_t.cpu()
    x_1 = x_1.cpu()
    max_val = torch.abs(x_1).max().item()

    # Determinants
    all_matrices = tf_t[:, :n_matrices]  # [T, 5, 2, 2]
    all_dets = torch.det(all_matrices).real  # [T, 5]
    det_min, det_max = all_dets.min().item(), all_dets.max().item()
    det_range = det_max - det_min
    det_min -= 0.1 * det_range
    det_max += 0.1 * det_range
    det_norm = Normalize(vmin=det_min, vmax=det_max)
    sm = cm.ScalarMappable(norm=det_norm, cmap=cm.plasma)

    for r in range(n_matrices):
        for c in range(nc):
            det = all_dets[c, r].item()
            ax_square = grid[r * nc + c]
            ax_square.set_aspect("equal")

            color = cm.plasma(det_norm(det))
            ax_square.plot(
                x_t[c, r, :, 0],
                x_t[c, r, :, 1],
                color=color,
                alpha=0.8,
                linewidth=2.5,
            )
            ax_square.plot(
                x_1[r, :, 0],
                x_1[r, :, 1],
                color="lightgray",
                alpha=0.8,
                linewidth=1.5,
                linestyle="--",
            )
            ax_square.set_xlim((-1.5 * max_val, 1.5 * max_val))
            ax_square.set_ylim((-1.5 * max_val, 1.5 * max_val))
            ax_square.set_xticks([])
            ax_square.set_yticks([])

            if r == 0 and titles is not None and c < len(titles):
                ax_square.set_title(titles[c], fontsize=18, pad=10)

    # --- Put colorbar in its own axis on the right ---
    cbar_ax = fig.add_axes([0.91, 0.2, 0.015, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Determinant", rotation=270, labelpad=20, fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # Adjust margins so titles aren’t cut
    plt.subplots_adjust(
        left=0.05,
        right=0.96,
        top=0.92,  # more room for titles
        bottom=0.06,
        hspace=0.0,
        wspace=0.0,
    )

    fig.canvas.draw()
    img_rgb = fig_to_img(fig)
    return img_rgb


def plot_x_t_centroids(x_t_all, t_values, titles=None):
    """Plot trajectories over time for matrix flow.

    Plot only centroids of x_t_all
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    t_vals = t_values.cpu().numpy()
    x_t_all = x_t_all.cpu().numpy()  # [T, B, N, 2]

    T, B = x_t_all.shape[:2]

    x_centroid = x_t_all.mean(axis=2)  # [T, B, 2]

    x_range = x_centroid[:, :, 0].max() - x_centroid[:, :, 0].min()
    shift_scale = x_range * 3

    x_centroid_shifted = x_centroid.copy()
    x_centroid_shifted[:, :, 0] += shift_scale * t_vals.reshape(-1, 1)

    ax.set_aspect("equal")
    ax.axis("off")

    y_max = x_centroid[:, :, 1].max()

    if titles is not None:
        ax.text(0, y_max * 1.3, titles[0], fontsize=16, ha="center", color="black")
        ax.text(
            shift_scale, y_max * 1.3, titles[1], fontsize=16, ha="center", color="black"
        )

    for i in range(B):
        # Plot trajectory as a line
        ax.plot(
            x_centroid_shifted[:, i, 0],
            x_centroid_shifted[:, i, 1],
            "-",
            linewidth=1,
            alpha=0.1,
            color="orange",
        )
        # Start and end points
        ax.plot(
            x_centroid_shifted[0, i, 0],
            x_centroid_shifted[0, i, 1],
            "o",
            color="blue",
            alpha=0.5,
            markersize=4,
        )
        ax.plot(
            x_centroid_shifted[-1, i, 0],
            x_centroid_shifted[-1, i, 1],
            "o",
            color="blue",
            alpha=0.5,
            markersize=4,
        )

    ax.set_xlabel("Time")

    fig.canvas.draw()
    img_rgb = fig_to_img(fig)

    return img_rgb


def create_animations(x_t, x_1, n_steps, titles, save_path):
    # x_t: [T+1, B, N, 2] tensor of predicted points at each time step
    # x_1: [B, N, 2] tensor of points of actual samples

    shift_min = np.array([-5, 0])
    shift_max = np.array([5, 0])

    B, N = x_1.shape[:2]

    # [T+1, 2]
    dots_shift = np.linspace(shift_min, shift_max, n_steps + 1)
    dots_shift = np.tile(dots_shift[:, None, None, :], (1, B, N, 1))  # [T+1, B, N, 2]

    # Source
    x_t_shifted = x_t.cpu().numpy() + dots_shift

    # Data
    x_1_shifted = x_1.cpu().numpy() + shift_max[None, None, :]

    # Set up the plot
    fig, axs = plt.subplots(B, 1, figsize=(10, 2 * B))
    if B == 1:
        axs = [axs]  # Ensure axs is iterable

    # Precompute interpolated paths for each sample
    times = np.linspace(0, 1, n_steps + 1)
    dummy_frames = n_steps // 10
    total_frames = n_steps + 2 * dummy_frames + 1

    # Compute the velocity of centroid as the difference between consecutive points / Δt
    Δt = 1.0 / n_steps  # Time step size
    velocities = np.zeros((n_steps + 1, B, 2))
    velocities[:-1] = np.diff(x_t.cpu().numpy().mean(2), axis=0) / Δt  # [T, B, 2]
    # Last velocity is zero

    # Pad dummy frames
    x_t_shifted = np.pad(
        x_t_shifted, [(dummy_frames, dummy_frames), (0, 0), (0, 0), (0, 0)], mode="edge"
    )
    velocities = np.pad(
        velocities, [(dummy_frames, dummy_frames), (0, 0), (0, 0)], mode="edge"
    )
    times = np.pad(times, dummy_frames, mode="edge")
    dots_shift = np.pad(
        dots_shift, [(dummy_frames, dummy_frames), (0, 0), (0, 0), (0, 0)], mode="edge"
    )

    # Initialize subplots
    flow_dots = []
    flow_velocities = []
    velocity_texts = []
    time_texts = []
    for i, ax in enumerate(axs):
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xlim(-6.2, 6.2)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        if i == 0:
            ax.text(
                shift_min[0], 1.4, titles[0], fontsize=12, ha="center", color="blue"
            )
            ax.text(
                shift_max[0], 1.4, titles[1], fontsize=12, ha="center", color="blue"
            )

        # Static source and target
        ax.scatter(
            x_t_shifted[0, i, :, 0], x_t_shifted[0, i, :, 1], c="blue", alpha=0.2, s=20
        )
        ax.scatter(
            x_1_shifted[i, :, 0], x_1_shifted[i, :, 1], c="blue", alpha=0.2, s=20
        )

        # Dynamic dot
        dot = ax.scatter(
            x_t_shifted[0, i, :, 0], x_t_shifted[0, i, :, 1], c="red", alpha=0.4, s=20
        )
        flow_dots.append(dot)

        # Dynamic quiver
        quiver = ax.quiver(
            x_t_shifted.mean(2)[0, i, 0],
            x_t_shifted.mean(2)[0, i, 1],
            velocities[0, i, 0],
            velocities[0, i, 1],
            angles="xy",
            scale_units="xy",
            scale=0.5,
            color="green",
            alpha=0.4,
        )
        quiver_text = ax.text(
            shift_min[0],
            1.1,
            rf"$\|v\|^2={np.linalg.norm(velocities[0, i]):.2f}$",
            fontsize=8,
            ha="center",
            color="green",
        )
        flow_velocities.append(quiver)
        velocity_texts.append(quiver_text)

        # Time text
        time_texts.append(
            ax.text(
                shift_min[0],
                -1.4,
                f"t={0:.2f}",
                fontsize=10,
                ha="center",
                color="black",
            )
        )

    # Update function for animation
    def update(frame_idx):
        artists = []

        for i, ax in enumerate(axs):
            flow_dots[i].set_offsets(x_t_shifted[frame_idx, i])
            flow_velocities[i].set_offsets(x_t_shifted.mean(2)[frame_idx, i])
            flow_velocities[i].set_UVC(
                velocities[frame_idx, i, 0], velocities[frame_idx, i, 1]
            )
            velocity_texts[i].set_text(
                rf"$\|v\|^2={np.linalg.norm(velocities[frame_idx, i]):.2f}$"
            )
            velocity_texts[i].set_x(dots_shift[frame_idx, i, 0, 0])

            time_texts[i].set_text(f"t={times[frame_idx]:.2f}")
            time_texts[i].set_x(dots_shift[frame_idx, i, 0, 0])

            artists.extend([flow_dots[i], flow_velocities[i], time_texts[i]])
        return artists

    ani = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=True)

    # Save animation
    ani.save(save_path, fps=5, dpi=100)


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg):
    log = get_logger(__name__)

    # Seed
    seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # Set up wandb and make checkpoints dir
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger = instantiate(
        cfg.logger, wandb_config=wandb_config, _convert_="all", _recursive_=False
    )

    # Datasets/Dataloaders
    train_dataset = instantiate(cfg.dataset.train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
    )
    test_dataset = instantiate(cfg.dataset.test)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )

    # Save dir
    (Path(cfg.save_dir) / "ckpt").mkdir(parents=True, exist_ok=True)

    # Model
    model = instantiate(cfg.model).to(cfg.device)

    # Training
    optimizer = instantiate(cfg.optimizer, model.parameters())

    # Optional resume from checkpoint. Only activates when model.checkpoint is set.
    checkpoint_dir = OmegaConf.select(cfg, "checkpoint", default=None)
    if checkpoint_dir:
        model_ckpt_path = Path(checkpoint_dir) / "model.pt"
        optimizer_ckpt_path = Path(checkpoint_dir) / "optimizer.pt"
        if not model_ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt_path}")
        if not optimizer_ckpt_path.exists():
            raise FileNotFoundError(
                f"Optimizer checkpoint not found: {optimizer_ckpt_path}"
            )

        model.load_state_dict(torch.load(model_ckpt_path, map_location=cfg.device))
        optimizer.load_state_dict(
            torch.load(optimizer_ckpt_path, map_location=cfg.device)
        )
        log.info(f"Loaded checkpoint from {checkpoint_dir}")
    
    # Learning rate scheduler (optional)
    scheduler = None
    if hasattr(cfg, 'scheduler') and cfg.scheduler is not None:
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    # Time indices (into the [0, n_steps] trajectory) at which to plot progression
    times = np.linspace(0, cfg.test.n_steps, 11)

    for epoch in trange(cfg.train.epochs, desc="Train"):
        # Train
        train_results = model.train_net(train_dataloader, optimizer)
        logger.log_all("train", train_results, {"epoch": epoch})
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_all("train", {"learning_rate": current_lr}, {"epoch": epoch})

        # Test
        test_results = model.eval_net(test_dataloader)
        logger.log_all("test", test_results, {"epoch": epoch})

        if (epoch + 1) % (cfg.test.epoch_interval) == 0:
            # Plot progression
            B = 5
            test_batch = sample_random_batch(test_dataset, B, cfg.device)

            # [T + 1, B, N, D]
            x_t, _, tf_t = model.sample_all(
                test_batch, cfg.test.n_steps, return_transform=True
            )
            x_t = x_t[times]
            tf_t = tf_t[times]

            titles = [f"t={t / cfg.test.n_steps}" for t in times]
            progression_img = plot_progression_grid(
                x_t, tf_t, test_batch, titles + ["Orig"]
            )
            logger.log_image(
                "test/progression",
                progression_img,
                {"epoch": epoch},
            )

            # Visualize random samples
            B = 64
            test_batch = sample_random_batch(test_dataset, B, cfg.device)

            test_samples = model.sample(
                test_batch, cfg.test.n_steps, return_transform=False
            )
            test_samples = test_samples.reshape(8, 8, *test_samples.shape[1:])

            samples_img = plot_2d_point_grid(test_samples.cpu().numpy())

            logger.log_image(
                "test/samples",
                samples_img,
                {"epoch": epoch},
            )

            # Check group_actions are close to C_4 elements
            B = 5_000
            test_batch, test_tf = sample_random_batch(test_dataset, B, cfg.device, return_transform=True)
            _, orig_tf, transforms = model.sample(
                test_batch, cfg.test.n_steps, return_transform=True
            )

            if model.prior_dist.group == "GL(2,C)":
                # Check for reflections (negative determinant)
                dets = torch.linalg.det(transforms)
                n_reflections = (dets.real < 0).sum().item()

                logger.log_all(
                    "test", {"reflection_ratio": n_reflections / B}, {"epoch": epoch}
                )

            # Extract angles
            # Convert real tensors to complex for compatibility with complex transforms
            if test_tf.is_complex() or orig_tf.is_complex() or transforms.is_complex():
                # Ensure all tensors are complex type
                if not test_tf.is_complex():
                    test_tf = torch.complex(test_tf, torch.zeros_like(test_tf))
                if not orig_tf.is_complex():
                    orig_tf = torch.complex(orig_tf, torch.zeros_like(orig_tf))
                if not transforms.is_complex():
                    transforms = torch.complex(transforms, torch.zeros_like(transforms))
            
            # canonicalized_transforms = (test_tf.adjoint() @ orig_tf.adjoint() @ transforms).adjoint()
            canonicalized_transforms = (orig_tf.adjoint() @ transforms).adjoint()
            
            # Check for NaN matrices and filter them out
            if torch.is_complex(canonicalized_transforms):
                # For complex tensors, check both real and imaginary parts
                nan_mask = torch.isnan(canonicalized_transforms.real).any(dim=(-2, -1)) | \
                          torch.isnan(canonicalized_transforms.imag).any(dim=(-2, -1)) | \
                          torch.isinf(canonicalized_transforms.real).any(dim=(-2, -1)) | \
                          torch.isinf(canonicalized_transforms.imag).any(dim=(-2, -1))
            else:
                # For real tensors
                nan_mask = torch.isnan(canonicalized_transforms).any(dim=(-2, -1)) | \
                          torch.isinf(canonicalized_transforms).any(dim=(-2, -1))
            
            nan_count = nan_mask.sum().item()
            total_count = nan_mask.shape[0]
            
            if nan_count > 0:
                # Filter out NaN matrices
                valid_mask = ~nan_mask
                canonicalized_transforms = canonicalized_transforms[valid_mask]
                # Also filter corresponding test_tf for downstream operations
                test_tf = test_tf[valid_mask]
                if canonicalized_transforms.shape[0] == 0:
                    continue
            
            R, P = polar_decomposition(canonicalized_transforms)
            angles = matrix_to_angle(R)
            angles = np.rad2deg(angles.cpu().numpy().flatten())
            angles_hist_img = plot_histogram(angles)
            logger.log_image(
                "test/angles_histogram",
                angles_hist_img,
                {"epoch": epoch},
            )
            # Plot the Frobenius norm of (P - I)
            I = torch.eye(P.shape[-1], device=cfg.device).unsqueeze(0)  # [1, D, D]
            P_norms = torch.linalg.norm(P - I, ord="fro", dim=(1, 2))
            total_count = canonicalized_transforms.shape[0]
            if torch.isinf(P_norms).any() or torch.isnan(P_norms).any():
                invalid_mask = torch.isinf(P_norms) | torch.isnan(P_norms)
                filtered_count = invalid_mask.sum().item()
                if filtered_count == total_count:
                    continue
                else:
                    canonicalized_transforms = canonicalized_transforms[~invalid_mask]
                    test_tf = test_tf[~invalid_mask]
                    P_norms = P_norms[~invalid_mask]
            mean_P_norms = P_norms.mean()
            logger.log_all(
                "test",
                {"norm(rem - I)": mean_P_norms.item()},
                {"epoch": epoch},
            )
            
            # Compute Wasserstein distance to prior distribution
            wasserstein_dist = compute_wasserstein_distance(
                canonicalized_transforms, test_tf
            )
            logger.log_all(
                "test",
                {"wasserstein_distance": wasserstein_dist},
                {"epoch": epoch},
            )

            # Plot trajectories over time
            B = 100
            test_batch = sample_random_batch(test_dataset, B, cfg.device)
            x_t_all = model.sample_all(test_batch, cfg.test.n_steps)
            t_values = torch.linspace(0, 1, cfg.test.n_steps + 1).to(cfg.device)

            traj_img = plot_x_t_centroids(
                x_t_all, t_values, [model.prior_dist.group, cfg.dataset.group]
            )
            logger.log_image(
                "test/prob_path",
                traj_img,
                {"epoch": epoch},
            )

            # Save model checkpoint
            torch.save(
                model.state_dict(),
                Path(cfg.save_dir) / "ckpt" / f"model.pt",
            )
            torch.save(
                optimizer.state_dict(),
                Path(cfg.save_dir) / "ckpt" / f"optimizer.pt",
            )

    # Create progression animation
    B = 5
    test_batch = sample_random_batch(test_dataset, B, cfg.device)
    x_t = model.sample_all(test_batch, cfg.test.n_steps)

    create_animations(
        x_t,
        test_batch,
        n_steps=cfg.test.n_steps,
        titles=[model.prior_dist.group, cfg.dataset.group],
        save_path=Path(cfg.save_dir) / "progression.mp4",
    )


if __name__ == "__main__":
    main()
