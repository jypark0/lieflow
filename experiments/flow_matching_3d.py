from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from hydra.utils import instantiate
from matplotlib.animation import FuncAnimation
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
import math

from lieflow.geometry_2d import polar_decomposition
from lieflow.geometry_3d import matrix_to_xyz_angles
from lieflow.logger import get_logger
from lieflow.utils import (
    fig_to_img,
    plot_histogram,
    sample_random_batch,
    seed_all,
)


class DataOnlyWrapper(Dataset):
    """
    Wrapper that only returns data (not transform) even if underlying dataset returns both.
    """
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        result = self.dataset[idx]
        # If result is a tuple (data, transform), return only data
        if isinstance(result, tuple):
            return result[0]
        return result

expm = torch.linalg.matrix_exp

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def compute_wasserstein_distance(
    learned_transforms, test_tf
):
    """
    Compute Wasserstein-1 distance between learned group elements and test_tf distribution.
    
    Treats each 3x3 matrix as a 9D vector in R^9 and computes the true Wasserstein-1
    distance using optimal transport (POT library).
    
    Args:
        learned_transforms: [B, 3, 3] learned transformation matrices
        test_tf: [B, 3, 3] test transformation matrices
        
    Returns:
        wasserstein_dist: Wasserstein-1 distance between matrix distributions
    """
    def so3_geodesic_cost_matrix(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
        """
        Pairwise SO(3) geodesic distances (in radians) between rotations.
        R_a: [N, 3, 3], R_b: [M, 3, 3]
        Returns: [N, M]
        """
        # Relative rotation: R_a^T R_b for all pairs
        rel = np.einsum("nik,mkj->nmij", R_a.transpose(0, 2, 1), R_b)  # [N, M, 3, 3]
        tr = np.einsum("nmii->nm", rel)  # [N, M]
        cos_theta = (tr - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)

    learned_matrices = learned_transforms.cpu().numpy()  # [B, 3, 3]
    n_learned = learned_matrices.shape[0]
    
    test_samples = test_tf.cpu().numpy()  # [B, 3, 3]
    n_test = test_samples.shape[0]
    
    # SO(3) geodesic cost matrix (radians), instead of Euclidean in R^9
    M = so3_geodesic_cost_matrix(learned_matrices, test_samples)  # [n_learned, n_test]
    
    a = np.ones(n_learned) / n_learned  # uniform distribution over learned samples
    b = np.ones(n_test) / n_test  # uniform distribution over test samples
    
    wasserstein_dist = ot.emd2(a, b, M)  # Earth Mover's Distance
    
    return wasserstein_dist


def plot_progression_grid(x_t, tf_t, x_1, titles=None):
    """
    Plot progression of 5 transformations on an irregular tetrahedron or hexahedron in 3D.
    Shows faces with fixed colors and black edges.
    Automatically detects whether object has 4 vertices (tetrahedron) or 8 vertices (hexahedron).
    No heatmaps, no per-row labels, no axes.
    Supports 4D data by automatically extracting first 3 dimensions for visualization.
    """
    n_matrices = 3
    nr, nc = n_matrices, x_t.shape[0]  # 3 rows, T columns
    fig = plt.figure(figsize=(1.5 * nc, 1.5 * nr), dpi=300)  # Reduced from 2.0

    # Move tensors to CPU
    x_t = x_t.cpu()
    tf_t = tf_t.cpu()
    x_1 = x_1.cpu()
    
    # Handle 4D data: extract first 3 dimensions for visualization
    if x_t.shape[-1] == 4:
        x_t = x_t[..., :3]
    if x_1.shape[-1] == 4:
        x_1 = x_1[..., :3]

    max_val = torch.abs(x_1).max().item()

    # Detect object type based on number of vertices
    n_vertices = x_1.shape[1]  # [B, N, 3] -> N
    
    if n_vertices == 3:
        # Triangle (identity point cloud: 3 points forming identity matrix)
        faces = [
            [0, 1, 2],  # Single triangle face
        ]
        face_colors = ["red"]
        vertex_colors = ["red", "green", "blue"]
    elif n_vertices == 4:
        # Tetrahedron faces (triangles)
        faces = [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ]
        face_colors = ["red", "green", "blue", "yellow"]
        vertex_colors = ["red", "green", "blue", "yellow"]
    elif n_vertices == 6:
        # Could be triangular prism (5 faces) or octahedron (8 faces)
        # Try to detect: if there's a vertex far from others (top/bottom), it's likely octahedron
        # For now, we'll use a simple heuristic based on z-coordinates
        # Check first sample to determine type
        sample_verts = x_1[0].detach().numpy() if hasattr(x_1[0], 'detach') else x_1[0]
        z_coords = sample_verts[:, 2]
        z_range = z_coords.max() - z_coords.min()
        z_mean = z_coords.mean()
        
        # If there are vertices clearly separated in z (top and bottom), it's likely octahedron
        # Otherwise, assume triangular prism
        has_top_bottom = (z_coords.max() > z_mean + 0.3 * z_range) and (z_coords.min() < z_mean - 0.3 * z_range)
        
        if has_top_bottom:
            # Octahedron: 8 triangular faces
            # Structure: top vertex (0), 4 middle vertices (1,2,3,4), bottom vertex (5)
            # Top pyramid: 0 with each pair of middle vertices
            # Bottom pyramid: 5 with each pair of middle vertices
            faces = [
                # Top pyramid (4 faces)
                [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
                # Bottom pyramid (4 faces)
                [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1],
            ]
            face_colors = ["red", "green", "blue", "yellow", "orange", "purple", "cyan", "magenta"]
            vertex_colors = ["red", "green", "blue", "yellow", "orange", "purple"]
        else:
            # Triangular prism: 5 faces
            # Bottom: 0,1,2; Top: 3,4,5
            faces = [
                [0, 1, 2],  # Bottom triangle
                [3, 4, 5],  # Top triangle
                [0, 1, 4, 3],  # Side 1
                [1, 2, 5, 4],  # Side 2
                [2, 0, 3, 5],  # Side 3
            ]
            face_colors = ["red", "green", "blue", "yellow", "orange"]
            vertex_colors = ["red", "green", "blue", "yellow", "orange", "purple"]
    elif n_vertices == 8:
        # Hexahedron faces (quadrilaterals)
        # Bottom: 0,1,2,3; Top: 4,5,6,7
        # Front: 0,1,5,4; Back: 2,3,7,6
        # Left: 0,3,7,4; Right: 1,2,6,5
        faces = [
            [0, 1, 2, 3],  # Bottom
            [4, 5, 6, 7],  # Top
            [0, 1, 5, 4],  # Front
            [2, 3, 7, 6],  # Back
            [0, 3, 7, 4],  # Left
            [1, 2, 6, 5],  # Right
        ]
        face_colors = ["red", "green", "blue", "yellow", "orange", "purple"]
        vertex_colors = ["red", "green", "blue", "yellow", "orange", "purple", "cyan", "magenta"]
    elif n_vertices == 20:
        # Dodecahedron (12 faces, each is a pentagon)
        # Simplified: use triangular faces connecting nearby vertices
        # For visualization, we'll create faces by connecting vertices in groups
        # This is a simplified representation
        faces = [
            # Top faces (simplified)
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
            # Middle faces
            [1, 5, 6], [2, 6, 7], [3, 7, 8], [4, 8, 5],
            [5, 9, 10], [6, 10, 11], [7, 11, 12], [8, 12, 9],
            # Bottom faces (simplified)
            [9, 13, 14], [10, 14, 15], [11, 15, 16], [12, 16, 13],
            [13, 17, 18], [14, 18, 19], [15, 19, 17], [16, 17, 18],
        ]
        face_colors = ["red", "green", "blue", "yellow", "orange", "purple", 
                      "cyan", "magenta", "pink", "brown", "gray", "olive",
                      "navy", "maroon", "teal", "lime", "coral", "gold"]
        vertex_colors = ["red", "green", "blue", "yellow", "orange", "purple", 
                        "cyan", "magenta", "pink", "brown", "gray", "olive",
                        "navy", "maroon", "teal", "lime", "coral", "gold", "silver", "indigo"]
    else:
        # Fallback: assume tetrahedron
        faces = [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ]
        face_colors = ["red", "green", "blue", "yellow"]
        vertex_colors = ["red", "green", "blue", "yellow"]

    for r in range(n_matrices):
        for c in range(nc):
            ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection="3d")

            # Get vertices and convert to numpy
            verts = x_t[c, r].detach().numpy().astype(float)

            # Create all faces as a single collection
            all_face_verts = []
            all_face_colors = []

            for face, fcolor in zip(faces, face_colors):
                face_verts = verts[face]
                all_face_verts.append(face_verts)
                all_face_colors.append(fcolor)

            # Create the main poly collection with faces but no edges initially
            poly_faces = Poly3DCollection(
                all_face_verts,
                facecolors=all_face_colors,
                alpha=0.6,
                edgecolors="none",  # Disable edges initially
                linewidths=0,
            )
            ax.add_collection3d(poly_faces)

            # Add black edges separately as line segments
            for face in faces:
                face_verts = verts[face]
                # Close the loop by adding the first vertex at the end
                edge_verts = np.vstack([face_verts, face_verts[0:1]])
                ax.plot(
                    edge_verts[:, 0],
                    edge_verts[:, 1],
                    edge_verts[:, 2],
                    color="black",
                    linewidth=1.0,
                )

            # Add colored vertices
            n_verts_to_plot = min(len(vertex_colors), len(verts))
            for v_idx in range(n_verts_to_plot):
                v_color = vertex_colors[v_idx % len(vertex_colors)]
                ax.scatter(
                    verts[v_idx, 0],
                    verts[v_idx, 1],
                    verts[v_idx, 2],
                    color=v_color,
                    s=30,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=10,
                )

            # Reference tetrahedron
            ref_verts = x_1[r].detach().numpy().astype(float)

            # Create reference faces with very transparent colors
            ref_face_verts = []
            ref_face_colors = []

            for face, fcolor in zip(faces, face_colors):
                face_verts = ref_verts[face]
                ref_face_verts.append(face_verts)
                ref_face_colors.append(fcolor)

            # Draw reference with very transparent colored faces
            ref_poly = Poly3DCollection(
                ref_face_verts,
                facecolors=ref_face_colors,
                alpha=0.1,  # Very transparent
                edgecolors="lightgray",
                linewidths=1.2,
                linestyles="--",
            )
            ax.add_collection3d(ref_poly)

            # Add colored vertices for reference object
            n_ref_verts_to_plot = min(len(vertex_colors), len(ref_verts))
            for v_idx in range(n_ref_verts_to_plot):
                v_color = vertex_colors[v_idx % len(vertex_colors)]
                ax.scatter(
                    ref_verts[v_idx, 0],
                    ref_verts[v_idx, 1],
                    ref_verts[v_idx, 2],
                    color=v_color,
                    s=40,
                    alpha=0.3,  # Make reference vertices more transparent
                    edgecolors="lightgray",
                    linewidths=0.5,
                    zorder=5,
                )

            # Set limits and aspect
            lim = 1.0 * max_val
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_box_aspect([1, 1, 1])

            # Viewing angle
            ax.view_init(elev=20, azim=45)

            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()

            # Column titles with minimal padding
            if r == 0 and titles is not None and c < len(titles):
                ax.set_title(titles[c], fontsize=18, pad=5)

    # Adjust layout with maximum negative spacing
    plt.subplots_adjust(
        left=0.00,
        right=1.00,
        top=0.92,  # Room for larger column titles
        bottom=0.00,
        hspace=-0.18,  # More negative spacing
        wspace=-0.18,  # More negative spacing
    )

    # Draw canvas
    fig.canvas.draw()
    img_rgb = fig_to_img(fig)

    return img_rgb


def plot_point_cloud_progression(x_t, tf_t, x_1, titles=None):
    """
    Plot progression of transformations on 3D point clouds.
    
    Args:
        x_t: [T, B, N, 3] tensor of point clouds at different time steps
        tf_t: [T, B, 3, 3] tensor of transformation matrices
        x_1: [B, N, 3] tensor of original point clouds
        titles: List of titles for each time step
    """
    n_samples = 3
    nr, nc = n_samples, x_t.shape[0]  # n_samples rows, T columns
    fig = plt.figure(figsize=(1.5 * nc, 1.5 * nr), dpi=150)
    
    # Move tensors to CPU
    x_t = x_t.cpu()
    tf_t = tf_t.cpu()
    x_1 = x_1.cpu()
    
    for r in range(n_samples):
        for c in range(nc):
            ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection="3d")
            
            # Get point cloud at this time step
            points = x_t[c, r].detach().numpy()  # [N, 3]
            orig_points = x_1[r].detach().numpy()  # [N, 3]
            
            # Plot transformed point cloud with fixed color
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c='blue',
                s=1,
                alpha=0.6,
            )
            
            # Plot original point cloud (transparent gray)
            ax.scatter(
                orig_points[:, 0],
                orig_points[:, 1],
                orig_points[:, 2],
                c='lightgray',
                s=0.5,
                alpha=0.3,
            )
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Set axis limits based on data range
            all_points = np.vstack([points, orig_points])
            max_range = np.array([
                all_points[:, 0].max() - all_points[:, 0].min(),
                all_points[:, 1].max() - all_points[:, 1].min(),
                all_points[:, 2].max() - all_points[:, 2].min()
            ]).max() / 2.0
            mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
            mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
            mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()
            
            # Set viewing angle
            ax.view_init(elev=20, azim=45)
            
            # Add title
            if r == 0 and titles is not None and c < len(titles):
                ax.set_title(titles[c], fontsize=18, pad=5)
    
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.05,
        hspace=0.2,
        wspace=0.2,
    )
    
    fig.canvas.draw()
    img_rgb = fig_to_img(fig)
    plt.close(fig)
    return img_rgb


def plot_point_cloud_grid(x, titles=None):
    """
    Plot grid of 3D point clouds.
    
    Args:
        x: [nr, nc, N, 3] numpy array of point clouds
        titles: Optional list of titles
    """
    nr, nc = x.shape[0], x.shape[1]
    fig = plt.figure(figsize=(1.5 * nc, 1.5 * nr), dpi=150)
    
    for i in range(nr * nc):
        row = i // nc
        col = i % nc
        ax = fig.add_subplot(nr, nc, i + 1, projection="3d")
        
        # Get point cloud
        points = x[row, col]  # [N, 3]
        
        # Plot point cloud
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c='blue',
            s=1,
            alpha=0.6,
        )
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set axis limits (finite coords only; NaN/Inf in samples poisons max/min)
        finite = np.isfinite(points).all(axis=1)
        pts = points[finite]
        if pts.size == 0:
            mid_x = mid_y = mid_z = 0.0
            max_range = 1.0
        else:
            spans = np.array(
                [
                    pts[:, 0].max() - pts[:, 0].min(),
                    pts[:, 1].max() - pts[:, 1].min(),
                    pts[:, 2].max() - pts[:, 2].min(),
                ]
            )
            max_range = float(spans.max()) / 2.0
            if not np.isfinite(max_range) or max_range == 0.0:
                max_range = 1.0
            mid_x = float((pts[:, 0].max() + pts[:, 0].min()) * 0.5)
            mid_y = float((pts[:, 1].max() + pts[:, 1].min()) * 0.5)
            mid_z = float((pts[:, 2].max() + pts[:, 2].min()) * 0.5)
            if not np.isfinite(mid_x):
                mid_x = 0.0
            if not np.isfinite(mid_y):
                mid_y = 0.0
            if not np.isfinite(mid_z):
                mid_z = 0.0
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Add title if provided
        if titles is not None and row == 0 and col < len(titles):
            ax.set_title(titles[col], fontsize=12, pad=5)
    
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.05,
        hspace=0.2,
        wspace=0.2,
    )
    
    fig.canvas.draw()
    img_rgb = fig_to_img(fig)
    plt.close(fig)
    return img_rgb


def plot_3D_scatter_grid(x, titles=None):
    """
    Plot grid of tetrahedrons or hexahedrons.
    Automatically detects object type based on number of vertices.
    Supports 4D data by automatically extracting first 3 dimensions for visualization.
    """
    nr, nc = x.shape[0], x.shape[1]
    fig = plt.figure(figsize=(1.5 * nc, 1.5 * nr), dpi=150)  # Reduced from 2.0
    
    # Handle 4D data: extract first 3 dimensions for visualization
    if x.shape[-1] == 4:
        x = x[..., :3]

    # Detect object type based on number of vertices (use first sample)
    n_vertices = x.shape[2]  # [nr, nc, N, 3] -> N
    
    if n_vertices == 3:
        # Triangle (identity point cloud: 3 points forming identity matrix)
        faces = [
            [0, 1, 2],  # Single triangle face
        ]
        face_colors = ["red"]
        vertex_colors = ["red", "green", "blue"]
    elif n_vertices == 4:
        # Tetrahedron faces (triangles)
        faces = [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ]
        face_colors = ["red", "green", "blue", "yellow"]
        vertex_colors = ["red", "green", "blue", "yellow"]
    elif n_vertices == 6:
        # Could be triangular prism or octahedron
        # Use first sample to detect
        sample_verts = x[0, 0]
        if hasattr(sample_verts, 'numpy'):
            sample_verts = sample_verts.numpy()
        z_coords = sample_verts[:, 2]
        z_range = z_coords.max() - z_coords.min()
        z_mean = z_coords.mean()
        has_top_bottom = (z_coords.max() > z_mean + 0.3 * z_range) and (z_coords.min() < z_mean - 0.3 * z_range)
        
        if has_top_bottom:
            # Octahedron: 8 triangular faces
            faces = [
                [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
                [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1],
            ]
            face_colors = ["red", "green", "blue", "yellow", "orange", "purple", "cyan", "magenta"]
            vertex_colors = ["red", "green", "blue", "yellow", "orange", "purple"]
        else:
            # Triangular prism: 5 faces
            faces = [
                [0, 1, 2], [3, 4, 5],
                [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5],
            ]
            face_colors = ["red", "green", "blue", "yellow", "orange"]
            vertex_colors = ["red", "green", "blue", "yellow", "orange", "purple"]
    elif n_vertices == 8:
        # Hexahedron faces (quadrilaterals)
        faces = [
            [0, 1, 2, 3],  # Bottom
            [4, 5, 6, 7],  # Top
            [0, 1, 5, 4],  # Front
            [2, 3, 7, 6],  # Back
            [0, 3, 7, 4],  # Left
            [1, 2, 6, 5],  # Right
        ]
        face_colors = ["red", "green", "blue", "yellow", "orange", "purple"]
        vertex_colors = ["red", "green", "blue", "yellow", "orange", "purple", "cyan", "magenta"]
    elif n_vertices == 20:
        # Dodecahedron (simplified faces for visualization)
        faces = [
            # Top faces
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
            # Middle faces
            [1, 5, 6], [2, 6, 7], [3, 7, 8], [4, 8, 5],
            [5, 9, 10], [6, 10, 11], [7, 11, 12], [8, 12, 9],
            # Bottom faces
            [9, 13, 14], [10, 14, 15], [11, 15, 16], [12, 16, 13],
            [13, 17, 18], [14, 18, 19], [15, 19, 17], [16, 17, 18],
        ]
        face_colors = ["red", "green", "blue", "yellow", "orange", "purple", 
                      "cyan", "magenta", "pink", "brown", "gray", "olive",
                      "navy", "maroon", "teal", "lime", "coral", "gold"]
        vertex_colors = ["red", "green", "blue", "yellow", "orange", "purple", 
                        "cyan", "magenta", "pink", "brown", "gray", "olive",
                        "navy", "maroon", "teal", "lime", "coral", "gold", "silver", "indigo"]
    else:
        # Fallback: assume tetrahedron
        faces = [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ]
        face_colors = ["red", "green", "blue", "yellow"]
        vertex_colors = ["red", "green", "blue", "yellow"]

    # Create subplots manually for 3D projection
    for i in range(nr * nc):
        row = i // nc
        col = i % nc
        # Create 3D subplot
        ax = fig.add_subplot(nr, nc, i + 1, projection="3d")

        # Get vertices
        verts = x[row, col].numpy() if hasattr(x[row, col], "numpy") else x[row, col]

        # Create all faces as a single collection
        all_face_verts = []
        all_face_colors = []

        for face, fcolor in zip(faces, face_colors):
            face_verts = verts[face]
            all_face_verts.append(face_verts)
            all_face_colors.append(fcolor)

        # Create the poly collection with colored faces
        poly_faces = Poly3DCollection(
            all_face_verts,
            facecolors=all_face_colors,
            alpha=0.6,
            edgecolors="none",
            linewidths=0,
        )
        ax.add_collection3d(poly_faces)

        # Add black edges separately as line segments
        for face in faces:
            face_verts = verts[face]
            # Close the loop by adding the first vertex at the end
            edge_verts = np.vstack([face_verts, face_verts[0:1]])
            ax.plot(
                edge_verts[:, 0],
                edge_verts[:, 1],
                edge_verts[:, 2],
                color="black",
                linewidth=1.0,
            )

        # Add colored vertices
        n_verts_to_plot = min(len(vertex_colors), len(verts))
        for v_idx in range(n_verts_to_plot):
            v_color = vertex_colors[v_idx % len(vertex_colors)]
            ax.scatter(
                verts[v_idx, 0],
                verts[v_idx, 1],
                verts[v_idx, 2],
                color=v_color,
                s=40,
                edgecolors="black",
                linewidths=0.5,
                zorder=10,
            )

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Set axis limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        # Remove all axes elements
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()

        # Set viewing angle
        ax.view_init(elev=20, azim=45)

        # Add title if provided (only for top row) with minimal padding
        if titles is not None and row == 0 and col < len(titles):
            ax.set_title(titles[col], fontsize=12, pad=-10)  # Reduced pad from 5 to 0

    # Adjust layout with maximum negative spacing
    plt.subplots_adjust(
        left=0.00,
        right=1.00,
        top=1.00,  # Very close to top, just leaving room for titles
        bottom=0.00,
        hspace=-0.18,  # More negative spacing
        wspace=-0.18,  # More negative spacing
    )

    fig.canvas.draw()
    img_rgb = fig_to_img(fig)

    return img_rgb


def plot_so3_distribution(
    rots: torch.Tensor,
    gt_rots=None,
    fig=None,
    ax=None,
    show_color_wheel: bool = True,
    canonical_rotation=torch.eye(3, dtype=torch.float32),
):
    """
    Map intrinsic xyz Euler angles to Mollweide projection.

    - X-axis: yaw angle (Z rotation)
    - Y-axis: pitch angle (Y rotation)
    - Color: roll angle (X rotation)
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 4), dpi=400)
        fig.subplots_adjust(left=0.10, bottom=0.12, right=0.90, top=0.95)
        ax = fig.add_subplot(111, projection="mollweide")

    cmap = plt.cm.hsv
    rots = rots.cpu()
    rots = rots @ canonical_rotation
    scatterpoint_scaling = 3  # Increased for visibility

    # Convert to XYZ Euler angles
    roll, pitch, yaw = matrix_to_xyz_angles(rots)

    # Add small random jitter to prevent exact overlaps
    jitter_scale = 0.02  # Adjust this to control jitter amount
    yaw_jittered = yaw + np.random.rand(len(yaw)) * jitter_scale
    pitch_jittered = pitch + np.random.rand(len(pitch)) * jitter_scale * 0.5

    # Color based on roll angle
    # Normalize roll from [-π, π] to [0, 1] for colormap
    colors = cmap((roll + np.pi) / (2.0 * np.pi))
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)
    colors[:, -1] = 0.8  # Set alpha to 0.6 for regular points

    # Display the distribution
    ax.scatter(
        yaw_jittered,
        pitch_jittered,
        s=scatterpoint_scaling,
        c=colors,
    )

    # Handle ground truth rotations if provided
    if gt_rots is not None:
        if gt_rots.dim() == 2:
            gt_rots = gt_rots.unsqueeze(0)
        gt_rots = gt_rots @ canonical_rotation

        # Get XYZ angles for ground truth
        roll_gt, pitch_gt, yaw_gt = matrix_to_xyz_angles(gt_rots)

        # Add jitter to GT points
        yaw_gt_jittered = yaw_gt + np.random.randn(len(yaw_gt)) * 3 * jitter_scale
        pitch_gt_jittered = (
            pitch_gt + np.random.randn(len(pitch_gt)) * 3 * jitter_scale * 0.5
        )

        # Color based on roll angle
        colors_gt = cmap((roll_gt + np.pi) / (2.0 * np.pi))
        if colors_gt.ndim == 1:
            colors_gt = colors_gt.reshape(1, -1)
        colors_gt[:, -1] = 1.0  # Full opacity for ground truth

        ax.scatter(
            yaw_gt_jittered,
            pitch_gt_jittered,
            s=60,  # Larger for ground truth
            c=colors_gt,
            edgecolors="black",
            marker="o",
            linewidth=1.0,
            zorder=10,
            alpha=0.2,
        )

    ax.grid()
    ax.set_xticklabels([])
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.tick_params(axis="both", which="minor", labelsize=8)

    # Add labels
    ax.set_xlabel("Yaw (Z rotation)", fontsize=12)
    ax.set_ylabel("Pitch (Y rotation)", fontsize=12)

    if show_color_wheel:
        # Add a color wheel showing the roll angle to color conversion
        ax_cw = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection="polar")
        theta = np.linspace(0, 2 * np.pi, 200)
        radii = np.linspace(0.4, 0.5, 2)
        theta_grid, _ = np.meshgrid(theta, radii)

        # Map the color wheel to show roll angle range [-π, π]
        colormap_val = theta_grid / (2 * np.pi)
        ax_cw.pcolormesh(theta, radii, colormap_val, cmap=cmap, shading="auto")
        ax_cw.set_yticklabels([])

        # Set tick labels for roll angles
        tick_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ax_cw.set_xticks(tick_angles)
        ax_cw.set_xticklabels(
            [
                "-180°",
                "-135°",
                "-90°",
                "-45°",
                "0°",
                "45°",
                "90°",
                "135°",
            ],
            fontsize=10,
        )
        ax_cw.spines["polar"].set_visible(False)

        # Add label for roll
        plt.text(
            0.5,
            0.5,
            "Roll (X)",
            fontsize=12,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax_cw.transAxes,
        )

    fig.canvas.draw()
    img = fig_to_img(fig)
    return img


def plot_x_t_centroids(x_t_all, t_values, titles=None):
    """Plot trajectories over time for matrix flow.

    Plot only centroids of x_t_all
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    t_vals = t_values.cpu().numpy()
    x_t_all = x_t_all.cpu().numpy()  # [T, B, N, 2]

    _, B = x_t_all.shape[:2]

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


def plot_entropy_p_x1_given_xt(results):
    p_x1_given_xt_all = results["p_x1_given_xt_all"].cpu()  # [n_timesteps, batch, M]
    t_values = results["t_values"].cpu()  # [n_timesteps]

    # Compute entropy for each time step
    eps = 1e-10
    p_safe = torch.clamp(p_x1_given_xt_all, min=eps)
    entropy_all = -(p_safe * torch.log(p_safe)).sum(dim=2)  # [n_timesteps, batch]

    entropy_mean = entropy_all.mean(dim=1).numpy()  # [n_timesteps]
    entropy_std = entropy_all.std(dim=1).numpy()  # [n_timesteps]

    t_values_np = t_values.numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean entropy
    ax.plot(t_values_np, entropy_mean, "b-", linewidth=2.5, label="Mean entropy")

    # Add confidence band
    ax.fill_between(
        t_values_np,
        entropy_mean - entropy_std,
        entropy_mean + entropy_std,
        alpha=0.3,
        color="blue",
        label="±1 std",
    )

    # Add max entropy line for reference (uniform distribution)
    n_components = p_x1_given_xt_all.shape[2]
    max_entropy = torch.log(torch.tensor(n_components)).item()
    ax.axhline(
        y=max_entropy,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Max entropy (uniform): {max_entropy:.2f}",
    )
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, label="Zero entropy")

    # Mark entropy at key points
    ax.scatter(
        [0],
        [entropy_mean[0]],
        color="green",
        s=80,
        zorder=5,
        label=f"t=0: {entropy_mean[0]:.3f}",
    )
    ax.scatter(
        [1],
        [entropy_mean[-1]],
        color="red",
        s=80,
        zorder=5,
        label=f"t=1: {entropy_mean[-1]:.3f}",
    )

    # Labels and formatting
    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Entropy H(p(x₁|xₜ)) [nats]", fontsize=12)
    ax.set_title("Mean Entropy of p(x₁|xₜ) Over Time", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Set y-limits with some padding
    y_min = min(0, entropy_mean.min() - 0.1)
    y_max = max(max_entropy + 0.1, entropy_mean.max() + entropy_std.max() + 0.1)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, 1)

    # Add text with batch size info
    ax.text(
        0.02,
        0.98,
        f"Batch size: {entropy_all.shape[1]} samples",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    fig.canvas.draw()
    img_rgb = fig_to_img(fig)

    return img_rgb


def create_animations(x_t, x_1, n_steps, titles, save_path):
    # x_t: [T+1, B, N, 2] tensor of predicted points at each time step
    # x_1: [B, N, 2] tensor of points of actual samples

    shift_min = np.array([-5, 0, 0])
    shift_max = np.array([5, 0, 0])

    B, N, D = x_1.shape

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
    velocities = np.zeros((n_steps + 1, B, D))
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
    # Check if we should use shared dataset (train/test split from same samples)
    if hasattr(cfg.dataset, 'share') and cfg.dataset.share:
        from torch.utils.data import Subset
        
        log.info("Using shared dataset mode: test is a subset of train samples")
        
        # Load full dataset with return_transform=True
        # We need to create a modified config that sets return_transform=True
        train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset.train, resolve=True))
        train_cfg.return_transform = True  # Enable transforms for both train and test
        full_dataset = instantiate(train_cfg)
        
        # Calculate split sizes
        train_ratio = cfg.dataset.get('train_ratio', 1.0)
        test_ratio = cfg.dataset.get('test_ratio', 0.5)
        
        total_size = len(full_dataset)
        train_size = int(total_size * train_ratio)
        test_size = int(total_size * test_ratio)
        
        # Create index ranges: test is the first test_size samples
        test_indices = list(range(test_size))
        train_indices = list(range(total_size))  # All samples for training
        
        # Create subsets
        train_subset = Subset(full_dataset, train_indices)
        test_subset = Subset(full_dataset, test_indices)
        
        # Wrap train dataset to only return data (not transform)
        train_dataset = DataOnlyWrapper(train_subset)
        test_dataset = test_subset  # Test returns (data, transform)
        
        log.info(f"Dataset split: {len(train_dataset)} train, {len(test_dataset)} test samples")
        log.info(f"Test set is the first {test_size} samples from training data")
        log.info(f"Train dataset: return_transform=False (wrapped)")
        log.info(f"Test dataset: return_transform=True")
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
        )
        
        # For shared dataset, copy dist attribute if it exists
        if hasattr(full_dataset, 'dist'):
            test_dataset.dataset.dist = full_dataset.dist
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
        )
    else:
        # Original behavior: separate train and test datasets
        train_dataset = instantiate(cfg.dataset.train)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
        )
        # Check if dataset has 'dist' field (for DistributionSamplingDataset)
        # Some datasets like ModelNet10 don't have this field
        if hasattr(cfg.dataset.train, 'dist') and cfg.dataset.train.dist is not None:
            if "Random" in cfg.dataset.train.dist.base_dist._target_:
                dist = train_dataset.dist
                test_dataset = instantiate(cfg.dataset.test, dist=dist)
            else:
                test_dataset = instantiate(cfg.dataset.test)
        else:
            # For datasets without 'dist' field
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

    # Optional resume from checkpoint. Only activates when checkpoint is set.
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
    times = np.linspace(0, cfg.test.n_steps, 7)

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
            B = 3
            test_batch = sample_random_batch(test_dataset, B, cfg.device)
            
            # Check if dataset is point cloud data by number of points.
            # Geometric primitives: tetrahedron (4), hexahedron (8).
            # Everything else (ModelNet10 with 128 pts, MI-Motion skeleton
            # with 20 joints, …) uses the scatter / point-cloud renderer.
            n_points = test_batch.shape[1]  # [B, N, D] -> N
            is_point_cloud = n_points > 8  # Threshold: >8 points = point cloud

            # [T + 1, B, N, D]
            x_t, _, tf_t = model.sample_all(
                test_batch, cfg.test.n_steps, return_transform=True
            )
            x_t = x_t[times]
            tf_t = tf_t[times]

            titles = [f"t={t / cfg.test.n_steps:.2f}" for t in times]
            
            if is_point_cloud:
                # Use point cloud visualization for ModelNet10
                progression_img = plot_point_cloud_progression(
                    x_t, tf_t, test_batch, titles + ["Orig"]
                )
            else:
                # Use tetrahedron visualization for other datasets
                progression_img = plot_progression_grid(
                    x_t, tf_t, test_batch, titles + ["Orig"]
                )
            
            logger.log_image(
                "test/progression",
                progression_img,
                {"epoch": epoch},
            )

            # Visualize random samples
            B = 48
            test_batch = sample_random_batch(test_dataset, B, cfg.device)

            test_samples = model.sample(
                test_batch, cfg.test.n_steps, return_transform=False
            )
            test_samples = test_samples.reshape(6, 8, *test_samples.shape[1:])

            if is_point_cloud:
                # Use point cloud visualization for ModelNet10
                samples_img = plot_point_cloud_grid(test_samples.cpu().numpy())
            else:
                # Use tetrahedron visualization for other datasets
                samples_img = plot_3D_scatter_grid(test_samples.cpu().numpy())

            logger.log_image(
                "test/samples",
                samples_img,
                {"epoch": epoch},
            )

            # Plot distribution
            B = 5_000
            test_batch, test_tf = sample_random_batch(
                test_dataset, B, cfg.device, return_transform=True
            )

            # If a ground-truth symmetry group is specified in the dataset config,
            # override test_tf with samples from that group so that W1 is computed
            # against the correct reference distribution.
            _gt_group = OmegaConf.select(cfg, "dataset.gt_group", default=None)
            if _gt_group == "SO2_z":
                # Uniform samples from SO(2): rotations around the z-axis
                _angles = torch.rand(B, device=cfg.device) * 2 * math.pi
                _c, _s = torch.cos(_angles), torch.sin(_angles)
                _z = torch.zeros(B, device=cfg.device)
                _o = torch.ones(B, device=cfg.device)
                test_tf = torch.stack([
                    torch.stack([_c, -_s, _z], dim=-1),
                    torch.stack([_s,  _c, _z], dim=-1),
                    torch.stack([_z,  _z, _o], dim=-1),
                ], dim=-2)  # [B, 3, 3]

            _, orig_tf, transforms = model.sample(
                test_batch, cfg.test.n_steps, return_transform=True
            )

            if model.prior_dist.group == "GL(3,C)":
                # Check for reflections (negative determinant)
                dets = torch.linalg.det(transforms)
                n_reflections = (dets.real < 0).sum().item()

                logger.log_all(
                    "test", {"reflection_ratio": n_reflections / B}, {"epoch": epoch}
                )

            # Extract angles
            # canonicalized_transforms = (
            #     test_tf.transpose(-2, -1) @ orig_tf.transpose(-2, -1) @ transforms
            # ).transpose(-2, -1)
            canonicalized_transforms = (
                orig_tf.transpose(-2, -1) @ transforms
            ).transpose(-2, -1)

            # Check for NaN/Inf matrices and filter (same idea as flow_matching_2d)
            if torch.is_complex(canonicalized_transforms):
                nan_mask = (
                    torch.isnan(canonicalized_transforms.real).any(dim=(-2, -1))
                    | torch.isnan(canonicalized_transforms.imag).any(dim=(-2, -1))
                    | torch.isinf(canonicalized_transforms.real).any(dim=(-2, -1))
                    | torch.isinf(canonicalized_transforms.imag).any(dim=(-2, -1))
                )
            else:
                nan_mask = torch.isnan(canonicalized_transforms).any(
                    dim=(-2, -1)
                ) | torch.isinf(canonicalized_transforms).any(dim=(-2, -1))

            nan_count = nan_mask.sum().item()
            total_count = nan_mask.shape[0]

            if nan_count > 0:
                valid_mask = ~nan_mask
                canonicalized_transforms = canonicalized_transforms[valid_mask]
                test_tf = test_tf[valid_mask]
                if canonicalized_transforms.shape[0] == 0:
                    continue

            # For 4D transforms, extract 3x3 rotation for visualization
            # but keep full 4x4 for other metrics
            canonicalized_transforms_3d = canonicalized_transforms
            if canonicalized_transforms.shape[-1] == 4:
                canonicalized_transforms_3d = canonicalized_transforms[..., :3, :3]
                
                # --- Extra logs for 4D case ---
                # 1) Determinant histogram of 3x3 sub-matrices
                det_values = torch.linalg.det(canonicalized_transforms_3d).real.cpu().numpy().flatten()
                # Filter out non-finite values (inf, -inf, nan) for histogram
                det_values_finite = det_values[np.isfinite(det_values)]
                if len(det_values_finite) == 0:
                    print("Warning: All determinant values are non-finite, skipping histogram")
                else:
                    fig_det, ax_det = plt.subplots(figsize=(6, 4))
                    ax_det.hist(det_values_finite, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
                    ax_det.set_xlabel("det(3x3)")
                    ax_det.set_ylabel("count")
                    ax_det.set_title("Determinant of 3x3 blocks (canonicalized)")
                    fig_det.canvas.draw()
                    det_img = fig_to_img(fig_det)
                    plt.close(fig_det)
                    logger.log_image(
                        "test/det3x3_hist",
                        det_img,
                        {"epoch": epoch},
                    )

                # 2) Mean canonicalized transform matrix (4x4) heatmap with values
                mean_tf = canonicalized_transforms.mean(dim=0).real.cpu().numpy()
                fig_mean, ax_mean = plt.subplots(figsize=(4.5, 4))
                im = ax_mean.imshow(mean_tf, cmap="coolwarm")
                ax_mean.set_title("Mean canonicalized transform (4x4)")
                ax_mean.set_xticks(range(mean_tf.shape[1]))
                ax_mean.set_yticks(range(mean_tf.shape[0]))
                for i in range(mean_tf.shape[0]):
                    for j in range(mean_tf.shape[1]):
                        ax_mean.text(
                            j,
                            i,
                            f"{mean_tf[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="black",
                        )
                fig_mean.colorbar(im, ax=ax_mean, fraction=0.046, pad=0.04)
                fig_mean.canvas.draw()
                mean_img = fig_to_img(fig_mean)
                plt.close(fig_mean)
                logger.log_image(
                    "test/canonicalized_mean_matrix",
                    mean_img,
                    {"epoch": epoch},
                )
            
            R, P = polar_decomposition(canonicalized_transforms_3d)
            
            # Get ground truth rotations
            # For datasets with dist attribute (e.g., DistributionSamplingDataset)
            if hasattr(test_dataset, 'dist') and test_dataset.dist is not None:
                base_dist = test_dataset.dist.base_dist
                # Check if base_dist has locs attribute (for discrete distributions)
                if hasattr(base_dist, 'locs'):
                    gt_rots = base_dist.locs
                    # For 4D ground truth, extract 3x3 for visualization
                    if gt_rots.shape[-1] == 4:
                        gt_rots = gt_rots[..., :3, :3]
                elif hasattr(base_dist, 'group') and base_dist.group == "SO(2)":
                    # For SO(2) continuous distribution, sample some rotations for visualization
                    # Sample uniformly from SO(2) to create a representative set
                    n_samples = 100
                    angles = torch.linspace(-torch.pi, torch.pi, n_samples)
                    c = torch.cos(angles)
                    s = torch.sin(angles)
                    zeros = torch.zeros_like(angles)
                    ones = torch.ones_like(angles)
                    gt_rots = torch.stack([
                        torch.stack([c, -s, zeros], dim=-1),
                        torch.stack([s, c, zeros], dim=-1),
                        torch.stack([zeros, zeros, ones], dim=-1),
                    ], dim=-2)  # [n_samples, 3, 3]
                elif hasattr(base_dist, 'group') and base_dist.group == "MultiAxisRotation":
                    # For MultiAxisRotation continuous distribution, sample rotations for each axis
                    # Generate samples for each rotation axis defined in the distribution
                    n_samples_per_axis = 100
                    angles = torch.linspace(-torch.pi, torch.pi, n_samples_per_axis)
                    
                    # Get rotation axes from the distribution
                    if base_dist.rotation_axes is not None:
                        rotation_axes = torch.tensor(base_dist.rotation_axes, dtype=torch.float32)  # [num_axes, 3]
                    else:
                        # Use default rotation axes: z-axis and two axes at ±60° to y-axis
                        rotation_axes = torch.tensor([
                            [0., 0., 1.],
                            [0., 0.5, math.sqrt(3)/2],
                            [0., 0.5, -math.sqrt(3)/2],
                        ], dtype=torch.float32)
                    num_axes = rotation_axes.shape[0]
                    
                    
                    # Generate rotations for each axis using Rodrigues' formula
                    all_gt_rots = []
                    for axis_idx in range(num_axes):
                        axis = rotation_axes[axis_idx]  # [3]
                        
                        # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
                        c = torch.cos(angles)  # [n_samples_per_axis]
                        s = torch.sin(angles)  # [n_samples_per_axis]
                        C = 1 - c  # [n_samples_per_axis]
                        
                        # Extract axis components
                        x, y, z = axis[0], axis[1], axis[2]
                        
                        # Construct rotation matrices using Rodrigues' formula
                        xs = x * s
                        ys = y * s
                        zs = z * s
                        xC = x * C
                        yC = y * C
                        zC = z * C
                        xyC = x * yC
                        yzC = y * zC
                        zxC = z * xC
                        
                        axis_rots = torch.stack([
                            torch.stack([x * xC + c, xyC - zs, zxC + ys], dim=-1),
                            torch.stack([xyC + zs, y * yC + c, yzC - xs], dim=-1),
                            torch.stack([zxC - ys, yzC + xs, z * zC + c], dim=-1),
                        ], dim=-2)  # [n_samples_per_axis, 3, 3]
                        
                        all_gt_rots.append(axis_rots)
                    
                    # Concatenate all rotations from different axes
                    gt_rots = torch.cat(all_gt_rots, dim=0)  # [num_axes * n_samples_per_axis, 3, 3] 
                else:
                    # For other continuous distributions, use Ico group as fallback
                    from lieflow.distributions import finite_group_elements_on_R3
                    gt_rots = finite_group_elements_on_R3("I")  # Ico as default
            else:
                # For datasets without dist, check cfg.dataset.gt_group
                _gt_group_rots = OmegaConf.select(cfg, "dataset.gt_group", default=None)
                if _gt_group_rots == "SO2_z":
                    # Dense circle of SO(2) z-axis rotations for 3D SO(3) plot reference
                    _n = 360
                    _angles = torch.linspace(-math.pi, math.pi, _n)
                    _c, _s = torch.cos(_angles), torch.sin(_angles)
                    _z = torch.zeros(_n)
                    _o = torch.ones(_n)
                    gt_rots = torch.stack([
                        torch.stack([_c, -_s, _z], dim=-1),
                        torch.stack([_s,  _c, _z], dim=-1),
                        torch.stack([_z,  _z, _o], dim=-1),
                    ], dim=-2)  # [360, 3, 3]
                else:
                    from lieflow.distributions import finite_group_elements_on_R3
                    gt_rots = finite_group_elements_on_R3("I")  # Ico as default
            
            # Apply conjugation for Oct group to align with a canonical representation
            # Conjugation: g @ R @ g^{-1}, where g is rotation of 90° around (1,1,1) axis
            is_oct_group = False
            if hasattr(test_dataset, 'dist') and test_dataset.dist is not None:
                base_dist = test_dataset.dist.base_dist
                # Check if it's Oct group by checking group attribute or locs size (24 elements)
                if hasattr(base_dist, 'group') and base_dist.group == "O":
                    is_oct_group = True
                elif hasattr(base_dist, 'locs') and base_dist.locs.shape[0] == 24:
                    # Could be Oct group if it has 24 elements
                    is_oct_group = True
            
            if is_oct_group:
                # Create rotation matrix g: 90° rotation around (1,1,1) axis
                # Normalize the axis
                axis = torch.tensor([1., 1., 1.], dtype=torch.float32, device=R.device)
                axis = axis / torch.norm(axis)  # normalize to unit vector
                
                # Rotation angle: 90 degrees = π/2
                theta = torch.tensor(math.pi / 2, dtype=torch.float32, device=R.device)
                
                # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
                # where K is the skew-symmetric matrix of the axis
                c = torch.cos(theta)
                s = torch.sin(theta)
                C = 1 - c
                
                x, y, z = axis[0], axis[1], axis[2]
                
                # Build rotation matrix using Rodrigues' formula
                g = torch.stack([
                    torch.stack([x*x*C + c,   x*y*C - z*s, x*z*C + y*s], dim=-1),
                    torch.stack([y*x*C + z*s, y*y*C + c,   y*z*C - x*s], dim=-1),
                    torch.stack([z*x*C - y*s, z*y*C + x*s, z*z*C + c  ], dim=-1),
                ], dim=-2)  # [3, 3]
                
                g_inv = g.transpose(-2, -1)  # g^{-1} for rotation matrices
                
                # Apply conjugation to R: g @ R @ g^{-1}
                R = torch.matmul(torch.matmul(g.unsqueeze(0), R), g_inv.unsqueeze(0))
                
                # Apply conjugation to gt_rots: g @ gt_rots @ g^{-1}
                # Move gt_rots to GPU for computation, then back to CPU
                gt_rots_device = gt_rots.device
                gt_rots_gpu = gt_rots.to(R.device)
                gt_rots_gpu = torch.matmul(torch.matmul(g.unsqueeze(0), gt_rots_gpu), g_inv.unsqueeze(0))
                gt_rots = gt_rots_gpu.to(gt_rots_device)

            # For the MI-Motion skeleton dataset there is no reliable ground-truth
            # symmetry, so we do not overlay the GT reference points on the plot.
            _dataset_name = OmegaConf.select(cfg, "dataset.name", default=None)
            _is_mi_motion = _dataset_name == "SO3_MI_Motion_skeleton"
            angles_hist_img = plot_so3_distribution(
                R, gt_rots=None if _is_mi_motion else gt_rots
            )
            logger.log_image(
                "test/learned_transforms_distribution",
                angles_hist_img,
                {"epoch": epoch},
            )
            
            # For SO(2) datasets (or datasets with gt_group="SO2_z"), also plot
            # angle histogram around z-axis to inspect the learned SO(2) structure.
            # The MI-Motion skeleton dataset is also expected to exhibit an
            # (approximate) SO(2) symmetry around the z-axis, so plot it there too.
            _is_so2_z = OmegaConf.select(cfg, "dataset.gt_group", default=None) == "SO2_z"
            if not _is_so2_z and hasattr(test_dataset, 'dist') and test_dataset.dist is not None:
                base_dist = test_dataset.dist.base_dist
                _is_so2_z = hasattr(base_dist, 'group') and base_dist.group == "SO(2)"
            if _is_so2_z or _is_mi_motion:
                # Extract SO(2) rotation angles from canonicalized_transforms.
                # For rotation around z-axis: R[1,0]=sin(θ), R[0,0]=cos(θ)
                angles_z = torch.atan2(
                    canonicalized_transforms[:, 1, 0],
                    canonicalized_transforms[:, 0, 0]
                )
                angles_z_deg = np.rad2deg(angles_z.cpu().numpy().flatten())
                so2_angles_hist_img = plot_histogram(angles_z_deg)
                logger.log_image(
                    "test/so2_angles_histogram",
                    so2_angles_hist_img,
                    {"epoch": epoch},
                )

            # Plot the Frobenius norm of (P - I)
            I = torch.eye(P.shape[-1], device=cfg.device).unsqueeze(0)  # [1, D, D]
            # P_norms = torch.linalg.norm(P - I, ord="fro", dim=(1, 2)).mean()
            P_norms = torch.linalg.norm(P - I, ord="fro", dim=(1, 2))
            total_count = canonicalized_transforms.shape[0]
            if torch.isinf(P_norms).any() or torch.isnan(P_norms).any():
                invalid_mask = torch.isinf(P_norms) | torch.isnan(P_norms)
                filtered_count = invalid_mask.sum().item()
                if filtered_count == total_count:
                    continue
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
            # Use full transformation matrices (4D or 3D)
            if not _is_mi_motion:
                wasserstein_dist = compute_wasserstein_distance(
                    canonicalized_transforms, test_tf
                )
                logger.log_all(
                    "test",
                    {"wasserstein_distance": wasserstein_dist},
                    {"epoch": epoch},
                )

            # ── Per-category evaluation for MultiSymmetryDataset ────────────
            # When the test dataset is a MultiSymmetryDataset (has .labels and
            # .dists), evaluate each symmetry category separately:
            # – plot an SO(3) distribution for each category (5 plots)
            # – compute a W1 distance for each category (5 values)
            if (
                hasattr(test_dataset, "labels")
                and hasattr(test_dataset, "dists")
                and test_dataset.transform is not None
            ):
                _cat_label_names = ["Tet", "Oct", "Ico", "SO2_z", "SO2_axis"]
                n_categories = len(test_dataset.dists)
                B_cat = 1_000  # samples per category

                for cat_idx in range(n_categories):
                    cat_mask = test_dataset.labels == cat_idx
                    cat_indices = cat_mask.nonzero(as_tuple=False).squeeze(-1)
                    if cat_indices.numel() == 0:
                        continue

                    n_cat = min(B_cat, cat_indices.numel())
                    rand_pos = torch.randint(cat_indices.numel(), (n_cat,))
                    rand_cat = cat_indices[rand_pos]

                    cat_batch = test_dataset.data[rand_cat].to(cfg.device)
                    cat_test_tf = test_dataset.transform[rand_cat].to(cfg.device)

                    # Run model on this category's samples
                    _, orig_tf_cat, transforms_cat = model.sample(
                        cat_batch, cfg.test.n_steps, return_transform=True
                    )

                    # Canonicalize: R_cat = (orig_tf^T @ transforms)^T
                    cat_canon = (
                        orig_tf_cat.transpose(-2, -1) @ transforms_cat
                    ).transpose(-2, -1)

                    # Filter NaN / Inf
                    nan_mask_cat = torch.isnan(cat_canon).any(
                        dim=(-2, -1)
                    ) | torch.isinf(cat_canon).any(dim=(-2, -1))
                    if nan_mask_cat.any():
                        cat_canon = cat_canon[~nan_mask_cat]
                        cat_test_tf = cat_test_tf[~nan_mask_cat]
                    if cat_canon.shape[0] == 0:
                        continue

                    # Polar decompose → SO(3) part for visualisation
                    R_cat, _ = polar_decomposition(cat_canon)

                    # Build ground-truth rotations for this category
                    cat_base_dist = test_dataset.dists[cat_idx].base_dist
                    if hasattr(cat_base_dist, "locs"):
                        # Finite group (Tet / Oct / Ico)
                        gt_rots_cat = cat_base_dist.locs
                    elif (
                        hasattr(cat_base_dist, "group")
                        and cat_base_dist.group == "SO(2)"
                    ):
                        # Continuous SO(2) around z-axis
                        _n_s = 100
                        _angles = torch.linspace(-math.pi, math.pi, _n_s)
                        _c, _s = torch.cos(_angles), torch.sin(_angles)
                        _z, _o = torch.zeros(_n_s), torch.ones(_n_s)
                        gt_rots_cat = torch.stack(
                            [
                                torch.stack([_c, -_s, _z], dim=-1),
                                torch.stack([_s, _c, _z], dim=-1),
                                torch.stack([_z, _z, _o], dim=-1),
                            ],
                            dim=-2,
                        )
                    elif (
                        hasattr(cat_base_dist, "group")
                        and cat_base_dist.group == "MultiAxisRotation"
                    ):
                        # Continuous SO(2) around arbitrary axis/axes
                        axes = cat_base_dist.rotation_axes.cpu()
                        _n_s = 100
                        _angles = torch.linspace(-math.pi, math.pi, _n_s)
                        _all_rots = []
                        for _ax in axes:
                            _x, _y, _z_ax = _ax[0], _ax[1], _ax[2]
                            _c = torch.cos(_angles)
                            _s = torch.sin(_angles)
                            _C = 1 - _c
                            _rots = torch.stack(
                                [
                                    torch.stack(
                                        [
                                            _x * _x * _C + _c,
                                            _x * _y * _C - _z_ax * _s,
                                            _x * _z_ax * _C + _y * _s,
                                        ],
                                        dim=-1,
                                    ),
                                    torch.stack(
                                        [
                                            _x * _y * _C + _z_ax * _s,
                                            _y * _y * _C + _c,
                                            _y * _z_ax * _C - _x * _s,
                                        ],
                                        dim=-1,
                                    ),
                                    torch.stack(
                                        [
                                            _x * _z_ax * _C - _y * _s,
                                            _y * _z_ax * _C + _x * _s,
                                            _z_ax * _z_ax * _C + _c,
                                        ],
                                        dim=-1,
                                    ),
                                ],
                                dim=-2,
                            )
                            _all_rots.append(_rots)
                        gt_rots_cat = torch.cat(_all_rots, dim=0)
                    else:
                        from lieflow.distributions import (
                            finite_group_elements_on_R3,
                        )
                        gt_rots_cat = finite_group_elements_on_R3("I")

                    cat_name = (
                        _cat_label_names[cat_idx]
                        if cat_idx < len(_cat_label_names)
                        else f"cat{cat_idx}"
                    )

                    # SO(3) distribution plot for this category
                    so3_img_cat = plot_so3_distribution(
                        R_cat, gt_rots=gt_rots_cat
                    )
                    logger.log_image(
                        f"test/so3_dist_cat{cat_idx}_{cat_name}",
                        so3_img_cat,
                        {"epoch": epoch},
                    )

                    # W1 distance for this category
                    w1_cat = compute_wasserstein_distance(
                        cat_canon, cat_test_tf
                    )
                    logger.log_all(
                        "test",
                        {f"w1_cat{cat_idx}_{cat_name}": w1_cat},
                        {"epoch": epoch},
                    )
            # ── End per-category evaluation ─────────────────────────────────

            # Plot trajectories over time
            B = 300
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

    # # Compute p(x_1 | x_t)
    # B = 500
    # test_batch = sample_random_batch(test_dataset, B, cfg.device)
    # x_0 = model.prior_dist.sample((B,)).to(cfg.device)
    # p1_dist = test_dataset.dist
    #
    # results = model.approx_p_x1_given_xt(x_0, p1_dist, n_steps=cfg.test.n_steps)
    #
    # entropy_p_x1_given_xt_img = plot_entropy_p_x1_given_xt(results)
    # logger.log_image(
    #     "test/entropy_p_x1_given_xt",
    #     entropy_p_x1_given_xt_img,
    #     {"epoch": epoch},
    # )
    #
    # # Create progression animation
    # B = 5
    # test_batch = sample_random_batch(test_dataset, B, cfg.device)
    # x_t = model.sample_all(test_batch, cfg.test.n_steps)
    #
    # create_animations(
    #     x_t,
    #     test_batch,
    #     n_steps=cfg.test.n_steps,
    #     titles=[model.prior_dist.base_dist.group, cfg.dataset.group],
    #     save_path=Path(cfg.save_dir) / "progression.mp4",
    # )


if __name__ == "__main__":
    main()
