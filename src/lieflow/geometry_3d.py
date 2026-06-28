import numpy as np
import torch






def matrix_to_xyz_angles(matrices):
    """
    Convert rotation matrices to intrinsic xyz Euler angles.

    Args:
        matrices: torch.Tensor of shape (..., 3, 3) rotation matrices

    Returns:
        roll, pitch, yaw: xyz Euler angles in radians
        - roll: rotation around x-axis [-π, π]
        - pitch: rotation around y-axis [-π/2, π/2]
        - yaw: rotation around z-axis [-π, π]
    """
    if not isinstance(matrices, torch.Tensor):
        matrices = torch.tensor(matrices, dtype=torch.float32)

    # Ensure we're working with float tensors
    matrices = matrices.float()

    # Extract elements
    r00 = matrices[..., 0, 0]
    r01 = matrices[..., 0, 1]
    r02 = matrices[..., 0, 2]
    r10 = matrices[..., 1, 0]
    r11 = matrices[..., 1, 1]
    r12 = matrices[..., 1, 2]
    r20 = matrices[..., 2, 0]
    r21 = matrices[..., 2, 1]
    r22 = matrices[..., 2, 2]

    # Pitch (Y rotation) - middle angle with limited range
    sin_pitch = -r20
    pitch = torch.asin(torch.clamp(sin_pitch, -1.0, 1.0))

    # Handle singularities when cos(pitch) ≈ 0
    cos_pitch = torch.cos(pitch)
    is_singular = torch.abs(cos_pitch) < 1e-6

    # Non-singular case
    roll = torch.atan2(r21 / cos_pitch, r22 / cos_pitch)
    yaw = torch.atan2(r10 / cos_pitch, r00 / cos_pitch)

    # Singular case (pitch ≈ ±π/2)
    # When pitch ≈ π/2: yaw - roll is determined
    # When pitch ≈ -π/2: yaw + roll is determined
    # We set roll = 0 and solve for yaw
    yaw_singular = torch.where(
        pitch > 0,
        torch.atan2(-r12, r11),  # pitch ≈ π/2
        torch.atan2(r12, r11),  # pitch ≈ -π/2
    )
    roll_singular = torch.zeros_like(pitch)

    # Select based on singularity
    roll = torch.where(is_singular, roll_singular, roll)
    yaw = torch.where(is_singular, yaw_singular, yaw)

    return roll, pitch, yaw






def quaternion_to_matrix(q):
    """
    Convert normalized quaternion q (..., 4) to rotation matrix (..., 3, 3)
    """
    w, x, y, z = q.unbind(-1)

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = ww - xx - yy + zz

    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )
