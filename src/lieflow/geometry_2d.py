import torch
from torch import Tensor


def angle_to_vector(theta: Tensor) -> Tensor:
    c = torch.cos(theta.squeeze())
    s = torch.sin(theta.squeeze())

    return torch.stack([c, s], dim=-1)  # [..., 2]


def angle_to_matrix(theta: Tensor) -> Tensor:
    c = torch.cos(theta.squeeze())
    s = torch.sin(theta.squeeze())
    return torch.stack(
        [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)],
        dim=-2,
    )  # [..., 2, 2]


def matrix_to_angle(mat: Tensor) -> Tensor:
    if torch.is_complex(mat):
        eigvals = torch.linalg.eigvals(mat)
        angles = torch.angle(eigvals)
        diff = angles[..., 1] - angles[..., 0]
        return diff.view(-1, 1)
    else:
        theta = torch.atan2(mat[..., 1, 0], mat[..., 0, 0])
        return theta.view(
            *mat.shape[:-2], 1
        )  # Keep the batch dimensions and add a new dimension for angle


def matrix_to_vector(mat: Tensor) -> Tensor:
    vec = mat[..., 0, :]  # Take the first column of the matrix
    return vec


def vector_to_angle(vec: Tensor) -> Tensor:
    theta = torch.atan2(vec[..., 1], vec[..., 0])
    return theta.view(-1, 1)


def wrap_angle(angles: Tensor) -> Tensor:
    """
    Wrap angles to the range [-pi, pi).
    """
    wrapped_angle = (angles + torch.pi) % (2 * torch.pi) - torch.pi
    return wrapped_angle


def polar_decomposition(mat):
    """
    Polar decomposition for complex or real matrices
    """
    # SVD returns W@diag(sigma)@V.T
    W, sigma, Vh = torch.linalg.svd(mat)

    sigma = torch.diag_embed(sigma).to(mat.dtype)
    V = Vh.adjoint()

    # Check if det(W @ V^T) < 0
    WV_adj = W @ V.adjoint()
    det = torch.det(WV_adj)

    # Construct a correction matrix D: identity with last diagonal = sign(det)
    D = (
        torch.eye(W.shape[-1], device=mat.device, dtype=mat.dtype)
        .expand_as(mat)
        .clone()
    )
    D[..., -1, -1] = torch.sgn(det)  # if det < 0, flip the sign

    R = torch.matmul(torch.matmul(W, D), Vh)
    P = torch.matmul(torch.matmul(V, sigma @ D), Vh)

    return R, P
