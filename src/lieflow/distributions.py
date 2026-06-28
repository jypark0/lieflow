import math

import numpy as np
import torch
import torch.distributions as td
from scipy.spatial.transform import Rotation

from lieflow.geometry_2d import (
    angle_to_matrix,
    angle_to_vector,
    matrix_to_angle,
    vector_to_angle,
    wrap_angle,
)
from lieflow.geometry_3d import quaternion_to_matrix
from lieflow.utils import blogm

expm = torch.linalg.matrix_exp


def Cn_elements_on_R2(group_order, representation="angle"):
    """
    Returns the elements of the cyclic group C_n in the given representation.

    Args:
        group_order (int): The order n of the cyclic group C_n.
        representation (str): One of "angle", "vector", or "matrix".

    Returns:
        torch.Tensor:
            - shape [n, 1] for "angle"
            - shape [n, 2] for "vector"
            - shape [n, 2, 2] for "matrix"
    """
    if group_order < 1:
        raise ValueError("group_order must be >= 1")

    # Generate angles in [0, 2π), then wrap to [-π, π)
    thetas = torch.arange(group_order) * (2 * torch.pi / group_order)  # [n]
    thetas = wrap_angle(thetas)  # Wrap angles to [-π, π)
    thetas = thetas.unsqueeze(-1)  # [n, 1]

    if representation == "angle":
        return thetas  # [n, 1]

    elif representation == "vector":
        return angle_to_vector(thetas)  # [n, 2]

    elif representation == "matrix":
        return angle_to_matrix(thetas)  # [n, 2, 2]

    else:
        raise ValueError(
            f"Invalid representation: {representation}. Must be one of 'angle',"
            " 'vector', or 'matrix'."
        )






def finite_group_elements_on_R3(group):
    elements = Rotation.create_group(group).as_matrix()

    return torch.tensor(elements, dtype=torch.float32)




# Ref: https://docs.pytorch.org/rl/main/reference/generated/torchrl.modules.Delta.html
class DiscreteDeltaMixture(td.Distribution):
    """
    Mixture of delta functions at discrete locations.
    Each sample returns one of the predefined locations with given probabilities.
    """

    arg_constraints = {}
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(
        self,
        group: str,
        locs: torch.Tensor,
        weights: torch.Tensor = None,
        validate_args=None,
    ):
        """
        Args:
            locs: [K, *event_shape] tensor of possible outcomes
            weights: [K] tensor of mixture weights, should sum to 1. If None, uniform.
        """
        self.group = group

        self.locs = locs  # [K, *event_shape]
        self.K = locs.shape[0]

        if weights is None:
            weights = torch.ones(self.K) / self.K
        if weights.sum() != 1:
            weights = weights / weights.sum()  # normalize to sum to 1
        self.weights = weights
        self.categorical = td.Categorical(probs=self.weights)

        event_shape = locs.shape[1:]
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def to(self, device):
        self.locs = self.locs.to(device)
        self.weights = self.weights.to(device)
        self.categorical = td.Categorical(probs=self.weights)

    def sample(self, sample_shape=torch.Size()):
        indices = self.categorical.sample(sample_shape)  # [sample_shape]
        samples = self.locs[indices]  # [*sample_shape, *event_shape]
        return samples

    def log_prob(self, value):
        """
        value: [..., *event_shape]
        Returns: [...], log probability of each value
        """
        # Reshape for broadcasting: [1, K, *event_shape] vs [..., 1, *event_shape]
        value_exp = value.unsqueeze(
            -self.locs.dim() - 1 + value.dim()
        )  # [..., 1, *event_shape]
        locs_exp = self.locs.unsqueeze(0)  # [1, K, *event_shape]
        matches = (value_exp == locs_exp).all(
            dim=(
                -1
                if self.event_shape == torch.Size()
                else tuple(range(-len(self.event_shape), 0))
            )
        )  # [..., K]

        # Weighted log prob
        log_weights = torch.log(self.weights)  # [K]
        log_probs = torch.where(
            matches, log_weights, torch.tensor(-float("inf"), device=log_weights.device)
        )  # [..., K]
        return torch.logsumexp(log_probs, dim=-1)  # sum over K


class RandomDiscreteDeltaMixture(DiscreteDeltaMixture):
    def __init__(
        self,
        num_modes: int,
        locs: torch.Tensor = None,
        validate_args=None,
    ):

        if locs is None:
            # Sample random locations on SO(3)
            locs = torch.randn(num_modes, 4)
            locs = locs / locs.norm(
                dim=-1, keepdim=True
            )  # Normalize to unit quaternions
            locs = quaternion_to_matrix(locs)  # Convert to rotation matrices [K, 3, 3]

        super().__init__(
            group=f"{num_modes} modes on SO(3)",
            locs=locs,
            weights=None,  # Uniform weights by default
            validate_args=validate_args,
        )






class SO2PushforwardDistribution(td.Distribution):
    arg_constraints = {}
    support = td.constraints.real  # Output is real matrices
    has_rsample = False

    def __init__(
        self,
        representation="matrix",
        coeff_dist=None,
        angle_range_deg=None,
        validate_args=False,
    ):
        """
        Pushforward distribution on SO(2) represented as 2x2 matrices via exponential map.

        Args:
            coeff_dist: Distribution over scalar coefficients of so(2) generator.
                        Default: Uniform(-pi, pi).
            angle_range_deg: Optional [low_deg, high_deg] pair (in degrees) to define
                             a uniform distribution over a sector of SO(2).
                             E.g. [-45, 45] gives Uniform(-π/4, π/4).
                             Takes precedence over coeff_dist.
        """
        self.group = "SO(2)"
        self.representation = representation
        if representation == "matrix":
            event_shape = (2, 2)
        elif representation == "vector":
            event_shape = (2,)
        elif representation == "angle":
            event_shape = (1,)
        else:
            raise NotImplementedError()

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=event_shape,
            validate_args=validate_args,
        )

        # Define Lie algebra basis of gl(2)
        self.basis = torch.tensor(
            [[0, -1], [1, 0]],
            dtype=torch.float32,
        )  # shape [2, 2]

        if angle_range_deg is not None:
            low = torch.tensor([math.radians(angle_range_deg[0])])
            high = torch.tensor([math.radians(angle_range_deg[1])])
            self.coeff_dist = td.Independent(td.Uniform(low, high), 1)
        elif coeff_dist is None:
            low = torch.tensor([-torch.pi])
            high = torch.tensor([torch.pi])
            self.coeff_dist = td.Independent(td.Uniform(low, high), 1)
        else:
            self.coeff_dist = coeff_dist

    def to(self, device):
        self.basis = self.basis.to(device)
        self.coeff_dist.base_dist.low = self.coeff_dist.base_dist.low.to(device)
        self.coeff_dist.base_dist.high = self.coeff_dist.base_dist.high.to(device)

    def sample(self, sample_shape=torch.Size()):
        theta = self.coeff_dist.sample(sample_shape)
        if self.representation == "angle":
            return theta
        elif self.representation == "vector":
            return angle_to_vector(theta)
        elif self.representation == "matrix":
            return angle_to_matrix(theta)
        else:
            raise NotImplementedError()

    def log_prob(self, x):
        """
        Evaluate log-probability under pushforward from angle base distribution.
        """
        if self.representation == "angle":
            theta = x
        elif self.representation == "vector":
            theta = vector_to_angle(x)
        elif self.representation == "matrix":
            theta = matrix_to_angle(x)
        else:
            raise NotImplementedError()

        return self.coeff_dist.log_prob(theta)


class SO2To3DDistribution(td.Distribution):
    """
    SO(2) distribution that outputs 3x3 rotation matrices for z-axis rotation.
    This is useful for applying SO(2) rotations to 3D objects.
    """
    arg_constraints = {}
    support = td.constraints.real  # Output is real matrices
    has_rsample = False

    def __init__(
        self,
        coeff_dist=None,
        validate_args=False,
    ):
        """
        Pushforward distribution on SO(2) represented as 3x3 matrices for z-axis rotation.

        Args:
            coeff_dist: Distribution over scalar coefficients (angles) of so(2) generator. 
                       Default: Uniform(-pi, pi)
        """
        self.group = "SO(2)"
        event_shape = (3, 3)

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=event_shape,
            validate_args=validate_args,
        )

        if coeff_dist is None:
            low = torch.tensor([-torch.pi])
            high = torch.tensor([torch.pi])
            self.coeff_dist = td.Independent(td.Uniform(low, high), 1)
        else:
            self.coeff_dist = coeff_dist

    def to(self, device):
        # Handle different types of base distributions
        if hasattr(self.coeff_dist, 'base_dist'):
            base_dist = self.coeff_dist.base_dist
            # For Uniform distribution
            if hasattr(base_dist, 'low') and hasattr(base_dist, 'high'):
                base_dist.low = base_dist.low.to(device)
                base_dist.high = base_dist.high.to(device)
            # For Normal distribution
            elif hasattr(base_dist, 'loc') and hasattr(base_dist, 'scale'):
                base_dist.loc = base_dist.loc.to(device)
                base_dist.scale = base_dist.scale.to(device)
        # If coeff_dist is directly a distribution (not wrapped in Independent)
        elif hasattr(self.coeff_dist, 'low') and hasattr(self.coeff_dist, 'high'):
            self.coeff_dist.low = self.coeff_dist.low.to(device)
            self.coeff_dist.high = self.coeff_dist.high.to(device)
        elif hasattr(self.coeff_dist, 'loc') and hasattr(self.coeff_dist, 'scale'):
            self.coeff_dist.loc = self.coeff_dist.loc.to(device)
            self.coeff_dist.scale = self.coeff_dist.scale.to(device)

    def _angle_to_3d_matrix(self, theta):
        """
        Convert SO(2) angle to 3x3 rotation matrix around z-axis.
        
        Args:
            theta: [..., 1] angle tensor
            
        Returns:
            [..., 3, 3] rotation matrix
        """
        theta = theta.squeeze(-1)  # [...,]
        c = torch.cos(theta)
        s = torch.sin(theta)
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)
        
        # Construct 3x3 rotation matrix around z-axis
        # R_z(theta) = [[cos(theta), -sin(theta), 0],
        #                [sin(theta),  cos(theta), 0],
        #                [0,           0,          1]]
        R = torch.stack([
            torch.stack([c, -s, zeros], dim=-1),  # [..., 3]
            torch.stack([s, c, zeros], dim=-1),   # [..., 3]
            torch.stack([zeros, zeros, ones], dim=-1),  # [..., 3]
        ], dim=-2)  # [..., 3, 3]
        
        return R

    def sample(self, sample_shape=torch.Size()):
        theta = self.coeff_dist.sample(sample_shape)  # [*sample_shape, 1]
        return self._angle_to_3d_matrix(theta)

    def log_prob(self, x):
        """
        Evaluate log-probability under pushforward from angle base distribution.
        
        Args:
            x: [..., 3, 3] rotation matrices
            
        Returns:
            log_prob: [...,] log probabilities
        """
        # Extract angle from 3x3 rotation matrix
        # For z-axis rotation: R[0,0] = cos(theta), R[1,0] = sin(theta)
        theta_hat = torch.atan2(x[..., 1, 0], x[..., 0, 0])  # [...,]
        theta_hat = theta_hat.unsqueeze(-1)  # [..., 1]
        
        return self.coeff_dist.log_prob(theta_hat)


class MultiAxisRotation3DDistribution(td.Distribution):
    """
    Distribution that outputs 3x3 rotation matrices by sampling rotations around multiple axes.
    Useful for applying rotations around multiple fixed axes to 3D objects.
    
    For each sample:
    1. Randomly select one of the provided rotation axes (uniformly)
    2. Sample a rotation angle from the angle distribution
    3. Return the 3x3 rotation matrix around the selected axis
    """
    arg_constraints = {}
    support = td.constraints.real  # Output is real matrices
    has_rsample = False

    def __init__(
        self,
        rotation_axes=None,
        coeff_dist=None,
        validate_args=False,
    ):
        """
        Distribution over rotations around multiple axes.

        Args:
            rotation_axes: List of 3D unit vectors representing rotation axes. 
                          Default: Three axes - z-axis, and two axes at ±60° to y-axis in yz-plane
            coeff_dist: Distribution over scalar coefficients (angles). 
                       Default: Uniform(-pi, pi)
        """
        self.group = "MultiAxisRotation"
        event_shape = (3, 3)

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=event_shape,
            validate_args=validate_args,
        )

        # Default rotation axes: z-axis and two axes at ±60° to y-axis
        if rotation_axes is None:
            # z-axis: [0, 0, 1]
            # axis at 60° to y-axis in yz-plane: [0, cos(60°), sin(60°)] = [0, 0.5, sqrt(3)/2]
            # axis at -60° to y-axis in yz-plane: [0, cos(-60°), sin(-60°)] = [0, 0.5, -sqrt(3)/2]
            self.rotation_axes = torch.tensor([
                # [0., 0., 1.],                              # z-axis
                # [0., 0.5, math.sqrt(3)/2],                 # +60° from y-axis
                [0., 0.5, -math.sqrt(3)/2],                # -60° from y-axis
            ], dtype=torch.float32)
        else:
            self.rotation_axes = torch.tensor(rotation_axes, dtype=torch.float32)
            
        # Normalize axes to unit vectors
        self.rotation_axes = self.rotation_axes / torch.norm(self.rotation_axes, dim=-1, keepdim=True)
        self.num_axes = self.rotation_axes.shape[0]

        # Angle distribution
        if coeff_dist is None:
            low = torch.tensor([-torch.pi])
            high = torch.tensor([torch.pi])
            self.coeff_dist = td.Independent(td.Uniform(low, high), 1)
        else:
            self.coeff_dist = coeff_dist

    def to(self, device):
        self.rotation_axes = self.rotation_axes.to(device)
        
        # Handle different types of base distributions
        if hasattr(self.coeff_dist, 'base_dist'):
            base_dist = self.coeff_dist.base_dist
            # For Uniform distribution
            if hasattr(base_dist, 'low') and hasattr(base_dist, 'high'):
                base_dist.low = base_dist.low.to(device)
                base_dist.high = base_dist.high.to(device)
            # For Normal distribution
            elif hasattr(base_dist, 'loc') and hasattr(base_dist, 'scale'):
                base_dist.loc = base_dist.loc.to(device)
                base_dist.scale = base_dist.scale.to(device)
        # If coeff_dist is directly a distribution (not wrapped in Independent)
        elif hasattr(self.coeff_dist, 'low') and hasattr(self.coeff_dist, 'high'):
            self.coeff_dist.low = self.coeff_dist.low.to(device)
            self.coeff_dist.high = self.coeff_dist.high.to(device)
        elif hasattr(self.coeff_dist, 'loc') and hasattr(self.coeff_dist, 'scale'):
            self.coeff_dist.loc = self.coeff_dist.loc.to(device)
            self.coeff_dist.scale = self.coeff_dist.scale.to(device)

    def _axis_angle_to_matrix(self, axis, theta):
        """
        Convert axis-angle representation to 3x3 rotation matrix using Rodrigues' formula.
        
        Args:
            axis: [..., 3] unit vector representing rotation axis
            theta: [...] rotation angle in radians
            
        Returns:
            [..., 3, 3] rotation matrix
        """
        # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        # where K is the skew-symmetric matrix of the axis
        
        theta = theta.squeeze(-1) if theta.dim() > axis.dim() - 1 else theta
        
        c = torch.cos(theta)  # [...]
        s = torch.sin(theta)  # [...]
        C = 1 - c  # [...]
        
        # Extract axis components
        x = axis[..., 0]  # [...]
        y = axis[..., 1]  # [...]
        z = axis[..., 2]  # [...]
        
        # Construct rotation matrix using Rodrigues' formula
        # More numerically stable formulation
        xs = x * s
        ys = y * s
        zs = z * s
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        
        R = torch.stack([
            torch.stack([x * xC + c, xyC - zs, zxC + ys], dim=-1),
            torch.stack([xyC + zs, y * yC + c, yzC - xs], dim=-1),
            torch.stack([zxC - ys, yzC + xs, z * zC + c], dim=-1),
        ], dim=-2)  # [..., 3, 3]
        
        return R

    def sample(self, sample_shape=torch.Size()):
        """
        Sample rotation matrices by randomly selecting an axis and angle.
        
        Returns:
            [..., 3, 3] rotation matrices
        """
        # Sample rotation angles
        theta = self.coeff_dist.sample(sample_shape)  # [*sample_shape, 1]
        
        # Randomly select axes (uniform over num_axes)
        total_samples = torch.Size(sample_shape).numel() if sample_shape else 1
        axis_indices = torch.randint(0, self.num_axes, (total_samples,), 
                                    device=self.rotation_axes.device)
        
        # Get selected axes
        selected_axes = self.rotation_axes[axis_indices]  # [total_samples, 3]
        
        # Reshape to match sample_shape
        if sample_shape:
            selected_axes = selected_axes.reshape(*sample_shape, 3)
        
        # Generate rotation matrices
        return self._axis_angle_to_matrix(selected_axes, theta)

    def log_prob(self, x):
        """
        Evaluate log-probability. Since we uniformly sample from multiple axes,
        this is a mixture distribution. For simplicity, we compute the log probability
        as if it came from any of the axes (mixture of uniforms).
        
        Args:
            x: [..., 3, 3] rotation matrices
            
        Returns:
            log_prob: [...] log probabilities
        """
        # For a mixture of K uniform distributions over angles on K axes,
        # the log probability is log(1/K) + log_prob_angle
        # We assume uniform mixture weights
        
        # Note: Extracting the exact axis and angle from a rotation matrix is non-trivial
        # For now, we return the base angle log prob minus log(num_axes) for the mixture
        
        # Extract rotation angle from matrix (axis-agnostic)
        # Using trace formula: tr(R) = 1 + 2*cos(theta)
        trace = x[..., 0, 0] + x[..., 1, 1] + x[..., 2, 2]  # [...]
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1, 1)  # Numerical stability
        theta = torch.acos(cos_theta)  # [...]
        theta = theta.unsqueeze(-1)  # [..., 1]
        
        # Base log prob from angle distribution
        angle_log_prob = self.coeff_dist.log_prob(theta)  # [...]
        
        # Account for mixture over axes (uniform mixture)
        mixture_log_prob = angle_log_prob - math.log(self.num_axes)
        
        return mixture_log_prob


class GL2PlusPushforwardDistribution(td.Distribution):
    arg_constraints = {}
    support = td.constraints.real
    has_rsample = False

    def __init__(
        self,
        coeff_dist=None,
        det_range=None,  # None means no restriction on determinants
        validate_args=False,
        estimate_acceptance_rate=True,
        n_samples_for_estimation=10_000,
    ):
        """
        Pushforward distribution on GL⁺(2) via exp from a uniform on Lie algebra coefficients.

        Args:
            coeff_dist: Distribution over coefficients of the Lie algebra gl(2). Default: Uniform([-1,-1,-1,-pi], [1,1,1,pi]).
        """
        self.group = "GL(2)"
        self.det_range = det_range

        event_shape = (2, 2)
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=event_shape,
            validate_args=validate_args,
        )

        # Standard basis for gl(2, R)
        self.basis = torch.tensor(
            [
                [[1, 0], [0, 0]],  # E_11
                [[0, 1], [0, 0]],  # E_12
                [[0, 0], [1, 0]],  # E_21
                [[0, 0], [0, 1]],  # E_22
            ],
            dtype=torch.float32,
        )  # shape [4, 2, 2]

        if coeff_dist is None:
            low = torch.tensor(
                [-torch.pi / 2, -torch.pi / 2, -torch.pi / 2, -torch.pi / 2],
            )
            high = torch.tensor(
                [torch.pi / 2, torch.pi / 2, torch.pi / 2, torch.pi / 2],
            )
            self.coeff_dist = td.Independent(td.Uniform(low, high), 1)  # [4]
        else:
            self.coeff_dist = coeff_dist

        # Estimate acceptance rate if requested
        if self.det_range is not None and estimate_acceptance_rate:
            self._estimate_acceptance_rate(n_samples_for_estimation)
        else:
            self.log_acceptance_rate = torch.tensor(0.0)

    def to(self, device):
        self.basis = self.basis.to(device)
        self.coeff_dist.base_dist.low = self.coeff_dist.base_dist.low.to(device)
        self.coeff_dist.base_dist.high = self.coeff_dist.base_dist.high.to(device)
        self.log_acceptance_rate = self.log_acceptance_rate.to(device)

    def _estimate_acceptance_rate(self, n_samples):
        """Estimate the acceptance rate for rejection sampling."""
        with torch.no_grad():
            # Sample many matrices from the unrestricted distribution
            coeffs = self.coeff_dist.sample((n_samples,))
            A = self._coeffs_to_matrix(coeffs)
            X = expm(A)

            # Check which ones satisfy the determinant constraint
            accepted = self._is_valid_det(X)

            # Estimate acceptance rate
            acceptance_rate = accepted.float().mean()

            # Add small epsilon to avoid log(0)
            eps = 1e-8
            self.log_acceptance_rate = torch.log(acceptance_rate + eps)

            print(f"Estimated acceptance rate: {acceptance_rate.item():.4f}")

    def _coeffs_to_matrix(self, coeffs):
        # coeffs: [..., 4], returns [..., 2, 2]
        return torch.einsum("...i,ijk->...jk", coeffs, self.basis)

    def _is_valid_det(self, X):
        """Check if matrices have determinant in the allowed range."""
        if self.det_range is None:
            return torch.linalg.det(X) > 0

        det = torch.linalg.det(X)
        return (det >= self.det_range[0]) & (det <= self.det_range[1])

    def sample(self, sample_shape=torch.Size(), max_attempts=1000):
        """Sample from GL^+(2)"""
        if self.det_range is None:
            # No restriction on determinants, just sample from the Lie algebra
            coeffs = self.coeff_dist.sample(sample_shape)  # [*sample_shape, 4]
            A = self._coeffs_to_matrix(coeffs)  # [*sample_shape, 2, 2]
            return expm(A)  # [*sample_shape, 2, 2]

        # Use rejection sampling to fit determinant range.
        total_shape = sample_shape + self.event_shape
        batch_size = torch.Size(sample_shape).numel()

        # Initialize output
        device = self.basis.device
        dtype = self.basis.dtype
        result = torch.empty(total_shape, device=device, dtype=dtype)

        # Track which samples still need to be filled
        remaining_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        flat_result = result.view(batch_size, *self.event_shape)

        attempts = 0
        while remaining_mask.any() and attempts < max_attempts:
            # Sample candidates for remaining positions
            n_remaining = remaining_mask.sum().item()
            coeffs = self.coeff_dist.sample((n_remaining,))
            A = self._coeffs_to_matrix(coeffs)
            X_candidates = expm(A)

            # Check which candidates are valid
            valid = self._is_valid_det(X_candidates)

            if valid.any():
                # Fill in valid samples
                remaining_indices = torch.where(remaining_mask)[0]
                valid_indices = torch.where(valid)[0]

                # Take as many valid samples as we can use
                n_to_fill = min(len(valid_indices), len(remaining_indices))
                flat_result[remaining_indices[:n_to_fill]] = X_candidates[
                    valid_indices[:n_to_fill]
                ]

                # Update remaining mask
                remaining_mask[remaining_indices[:n_to_fill]] = False

            attempts += 1

        if remaining_mask.any():
            print(
                "Warning: Could not generate all samples after"
                f" {max_attempts} attempts. {remaining_mask.sum().item()} samples"
                " remaining."
            )
            # Fill remaining with identity matrices (valid since det(I) = 1 ∈ [0.5, 2])
            I = torch.eye(2, device=device, dtype=dtype)
            flat_result[remaining_mask] = I

        return result

    def project_to_manifold(self, X: torch.Tensor, eps=1e-6) -> torch.Tensor:
        """Project matrices to GL^+(2) by ensuring positive determinant."""

        det = torch.linalg.det(X)

        # For negative determinants, flip sign of one column
        negative_mask = det < 0
        if negative_mask.any():
            X = X.clone()
            X[negative_mask, :, 0] *= -1

        # For near-zero determinants, add small identity
        small_det_mask = torch.abs(torch.linalg.det(X)) < eps
        if small_det_mask.any():
            X = X.clone()
            I = torch.eye(2, device=X.device, dtype=X.dtype)
            X[small_det_mask] += eps * I

        return X

    def log_prob(self, X):
        """
        Exact log-probability for X under the pushforward from Lie algebra,
        including the Jacobian determinant of exp, restricted to GL⁺(2) (det > 0).

        Args:
            X: Tensor of shape [..., 2, 2]

        Returns:
            Tensor of shape [...] with log-probabilities, or -inf where det(X) <= 0.
        """

        # Batch shape
        batch_shape = X.shape[:-2]
        device = X.device
        dtype = X.dtype

        # Check positive determinant (GL⁺(2) support)
        det = torch.linalg.det(X)
        if self.det_range is None:
            valid_mask = det > 0
        else:
            valid_mask = (det >= self.det_range[0]) & (det <= self.det_range[1])

        # Initialize result with -inf
        result = torch.full(batch_shape, float("-inf"), device=device, dtype=dtype)
        if not valid_mask.any():
            return result  # all invalid

        # Extract valid X entries
        X_valid = X[valid_mask]  # [N_valid, 2, 2]

        # Compute matrix logarithm for valid entries
        A = blogm(X_valid)  # [N_valid, 2, 2]

        # Project onto basis to get coefficients
        # Normalize by squared Frobenius norms of the basis
        coeffs = torch.einsum("...ij,kij->...k", A, self.basis)  # [N_valid, 4]

        # Clamp the first three coefficients to [-1, 1] for stability
        low = self.coeff_dist.base_dist.low[:3].to(device, dtype)
        high = self.coeff_dist.base_dist.high[:3].to(device, dtype)
        coeffs[:, :3] = coeffs[:, :3].clamp(low, high)
        # Wrap the rotation coefficient to [-pi, pi]
        coeffs[:, 3] = wrap_angle(coeffs[:, 3])  # [N_valid, 4]
        log_p_base = self.coeff_dist.log_prob(coeffs)  # [N_valid]

        # Prepare for Jacobian: compute flattened A from coeffs
        # flattened A = sum_i coeffs[i] * basis[i].flatten
        A_flat = coeffs @ self.basis.view(4, -1)  # [N_valid, 4]
        A_flat = A_flat.detach().requires_grad_(True)

        # Define function from flattened coeffs to flattened exp(A)
        def exp_from_flat(coeffs_flat):
            # coeffs_flat: [N_valid, 4]
            # Reconstruct A flattened then matrix form
            A_batch = coeffs_flat @ self.basis.view(4, -1)  # [N_valid, 4]
            A_batch = A_batch.view(-1, 2, 2)  # [N_valid, 2, 2]
            expA = expm(A_batch)  # [N_valid, 2, 2]
            return expA.reshape(-1, 4)  # [N_valid, 4]

        # Compute Jacobian: shape [N_valid, 4, N_valid, 4]
        # Use vectorize=True for efficiency if available
        jac = torch.autograd.functional.jacobian(
            exp_from_flat, A_flat, create_graph=False, vectorize=True
        )  # [N_valid, 4, N_valid, 4]

        # Extract per-sample Jacobians: [N_valid, 4, 4]
        # Permute to [N_valid, N_valid, 4, 4], then take diagonal along samples
        J = (
            jac.permute(2, 0, 1, 3).diagonal(dim1=0, dim2=1).permute(2, 0, 1)
        )  # [N_valid, 4, 4]

        # Compute log|det(J)| robustly
        eps = 1e-12
        eye = torch.eye(4, device=device, dtype=dtype)
        # Use abs to handle sign, though for exp Jacobian det > 0 often
        _, log_abs_det = torch.linalg.slogdet(J.abs() + eps * eye)  # [N_valid]

        # Assign valid entries
        if self.det_range is None:
            result_flat = log_p_base - log_abs_det  # [N_valid]
        else:
            # For restricted range, subtract log acceptance rate
            result_flat = (
                log_p_base - log_abs_det - self.log_acceptance_rate
            )  # [N_valid]

        result[valid_mask] = result_flat

        return result






class GL2RotationScaleShearDistribution(td.Distribution):
    """
    GL(2) distribution using Rotation-Scale-Shear decomposition:
    X = R @ S @ Sh
    where:
    - R ~ SO(2): rotation matrix
    - S: diagonal scale matrix [[s_x, 0], [0, s_y]]
    - Sh: upper triangular shear matrix [[1, k], [0, 1]]
    
    det(X) = det(R) × det(S) × det(Sh) = 1 × (s_x × s_y) × 1 = s_x × s_y
    """
    arg_constraints = {}
    support = td.constraints.real
    has_rsample = False

    def __init__(
        self,
        scale_range=(0.5, 2.0),
        shear_range=(-1.0, 1.0),
        det_range=(-2.0, 2.0),
        allow_negative_det=True,
        prob_negative_det=0.5,
        scale_std=0.3,
        shear_std=0.3,
        validate_args=False,
    ):
        """
        Rotation-Scale-Shear decomposition distribution on GL(2).
        
        Args:
            scale_range: Tuple (min_scale, max_scale) for scale factors s_x, s_y.
                        Default: (0.5, 2.0) - used for truncation bounds
            shear_range: Tuple (min_shear, max_shear) for shear parameter k.
                        Default: (-1.0, 1.0) - used for truncation bounds
            det_range: Tuple (min_det, max_det) for determinant constraint.
                      Default: (-2, 2)
            allow_negative_det: If True, allows negative determinants via reflection.
                               Default: True
            prob_negative_det: Probability of negative determinant (if allowed).
                              Default: 0.5
            scale_std: Standard deviation for scale Gaussian distribution (mean=1.0).
                      Default: 0.3
            shear_std: Standard deviation for shear Gaussian distribution (mean=0.0).
                      Default: 0.3
        """
        self.group = "GL(2)"
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.det_range = det_range
        self.allow_negative_det = allow_negative_det
        self.prob_negative_det = prob_negative_det if allow_negative_det else 0.0
        self.scale_std = scale_std
        self.shear_std = shear_std

        event_shape = (2, 2)
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=event_shape,
            validate_args=validate_args,
        )

        # Distribution for SO(2) rotation angles
        angle_low = torch.tensor([-torch.pi])
        angle_high = torch.tensor([torch.pi])
        self.angle_dist = td.Independent(td.Uniform(angle_low, angle_high), 1)

        # Distribution for scale factors [s_x, s_y] - Uniform distribution
        scale_low = torch.tensor([scale_range[0], scale_range[0]])
        scale_high = torch.tensor([scale_range[1], scale_range[1]])
        self.scale_dist = td.Independent(td.Uniform(scale_low, scale_high), 1)
        # Store bounds for reference (determinant constraint will handle filtering)
        self.scale_low = scale_low
        self.scale_high = scale_high

        # Distribution for shear parameter k - Gaussian with mean=0.0
        shear_mean = torch.tensor([0.0])
        shear_std_tensor = torch.tensor([shear_std])
        shear_low = torch.tensor([shear_range[0]])
        shear_high = torch.tensor([shear_range[1]])
        self.shear_dist = td.Independent(td.Normal(shear_mean, shear_std_tensor), 1)
        # Store bounds for manual truncation if needed
        self.shear_low = shear_low
        self.shear_high = shear_high

        # Reflection matrix for negative determinants
        self.reflection_matrix = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)

    def to(self, device):
        self.angle_dist.base_dist.low = self.angle_dist.base_dist.low.to(device)
        self.angle_dist.base_dist.high = self.angle_dist.base_dist.high.to(device)
        self.scale_dist.base_dist.low = self.scale_dist.base_dist.low.to(device)
        self.scale_dist.base_dist.high = self.scale_dist.base_dist.high.to(device)
        self.shear_dist.base_dist.loc = self.shear_dist.base_dist.loc.to(device)
        self.shear_dist.base_dist.scale = self.shear_dist.base_dist.scale.to(device)
        self.scale_low = self.scale_low.to(device)
        self.scale_high = self.scale_high.to(device)
        self.shear_low = self.shear_low.to(device)
        self.shear_high = self.shear_high.to(device)
        self.reflection_matrix = self.reflection_matrix.to(device)

    def _angle_to_rotation(self, theta):
        """Convert angle to SO(2) rotation matrix."""
        theta = theta.squeeze(-1)  # [...,]
        c = torch.cos(theta)
        s = torch.sin(theta)
        return torch.stack(
            [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)],
            dim=-2,
        )  # [..., 2, 2]

    def _construct_scale_matrix(self, scales):
        """
        Construct diagonal scale matrix from scale factors.
        S = [[s_x, 0], [0, s_y]]
        """
        return torch.diag_embed(scales)  # [..., 2, 2]

    def _construct_shear_matrix(self, shear):
        """
        Construct upper triangular shear matrix.
        Sh = [[1, k], [0, 1]]
        """
        shear = shear.squeeze(-1)  # [...,]
        ones = torch.ones_like(shear)
        zeros = torch.zeros_like(shear)
        
        return torch.stack(
            [torch.stack([ones, shear], dim=-1), torch.stack([zeros, ones], dim=-1)],
            dim=-2,
        )  # [..., 2, 2]

    def _is_valid_det(self, X):
        """Check if matrices have determinant in the allowed range."""
        det = torch.linalg.det(X)
        return (det >= self.det_range[0]) & (det <= self.det_range[1])

    def sample(self, sample_shape=torch.Size(), max_attempts=1000):
        """
        Sample from GL(2) using Rotation-Scale-Shear decomposition:
        X = (Reflection) @ R @ S @ Sh
        """
        batch_size = torch.Size(sample_shape).numel()
        
        # Sample rotation angles
        theta = self.angle_dist.sample(sample_shape)  # [*sample_shape, 1]
        R = self._angle_to_rotation(theta)  # [*sample_shape, 2, 2]

        device = R.device
        dtype = R.dtype
        
        # Determine which samples should have negative determinant
        if self.allow_negative_det:
            use_reflection = torch.rand(batch_size, device=device) < self.prob_negative_det
        else:
            use_reflection = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Initialize result storage
        total_shape = sample_shape + self.event_shape
        result = torch.empty(total_shape, device=device, dtype=dtype)
        
        # Track which samples still need to be filled
        remaining_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        flat_result = result.view(batch_size, *self.event_shape)
        flat_R = R.view(batch_size, *self.event_shape)

        attempts = 0
        while remaining_mask.any() and attempts < max_attempts:
            # Sample scale and shear for remaining positions
            n_remaining = remaining_mask.sum().item()
            scales = self.scale_dist.sample((n_remaining,))  # [n_remaining, 2]
            shears = self.shear_dist.sample((n_remaining,))  # [n_remaining, 1]
            
            S_candidates = self._construct_scale_matrix(scales)  # [n_remaining, 2, 2]
            Sh_candidates = self._construct_shear_matrix(shears)  # [n_remaining, 2, 2]

            # Get corresponding R matrices
            remaining_indices = torch.where(remaining_mask)[0]
            R_remaining = flat_R[remaining_indices]  # [n_remaining, 2, 2]
            
            # Apply reflection for those that need it
            use_reflection_remaining = use_reflection[remaining_indices]
            Refl_remaining = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(n_remaining, 2, 2).clone()
            Refl_remaining[use_reflection_remaining] = self.reflection_matrix

            # Compute X = Reflection @ R @ S @ Sh
            X_candidates = Refl_remaining @ R_remaining @ S_candidates @ Sh_candidates

            # Check which candidates satisfy determinant constraint
            valid = self._is_valid_det(X_candidates)

            if valid.any():
                # Fill in valid samples
                valid_indices = torch.where(valid)[0]

                # Take as many valid samples as we can use
                n_to_fill = min(len(valid_indices), len(remaining_indices))
                flat_result[remaining_indices[:n_to_fill]] = X_candidates[
                    valid_indices[:n_to_fill]
                ]

                # Update remaining mask
                remaining_mask[remaining_indices[:n_to_fill]] = False

            attempts += 1

        if remaining_mask.any():
            print(
                f"Warning: Could not generate all samples after {max_attempts} attempts. "
                f"{remaining_mask.sum().item()} samples remaining."
            )
            # Fill remaining with identity matrices
            I = torch.eye(2, device=device, dtype=dtype)
            flat_result[remaining_mask] = I

        return result

    def log_prob(self, X):
        """
        Compute log probability.
        
        For now, returns uniform log prob (placeholder for proper implementation).
        """
        batch_shape = X.shape[:-2]
        device = X.device
        dtype = X.dtype

        # Check determinant constraint
        valid_mask = self._is_valid_det(X)

        # Initialize result with -inf
        result = torch.full(batch_shape, float("-inf"), device=device, dtype=dtype)
        
        if not valid_mask.any():
            return result

        # For valid samples, compute uniform probability over the constrained space
        X_valid = X[valid_mask]
        
        # Placeholder: uniform log probability
        log_p = torch.zeros(X_valid.shape[0], device=device, dtype=dtype)
        
        result[valid_mask] = log_p
        return result

    def decompose(self, X):
        """
        Decompose a GL(2) matrix into Rotation-Scale-Shear components.
        Returns rotation angle, scale factors, and shear parameter.
        
        Note: This is an approximation via QR decomposition.
        """
        # QR decomposition: X = Q @ R_upper
        # Q is orthogonal (rotation possibly with reflection)
        # R_upper is upper triangular
        Q, R_upper = torch.linalg.qr(X)
        
        # Extract rotation angle from Q
        # Handle possible reflection in Q
        det_Q = torch.linalg.det(Q)
        theta = torch.atan2(Q[..., 1, 0], Q[..., 0, 0])
        
        # Extract scale and shear from R_upper
        # R_upper = [[r11, r12], [0, r22]]
        s_x = R_upper[..., 0, 0]
        s_y = R_upper[..., 1, 1]
        k = R_upper[..., 0, 1] / s_x  # shear parameter
        
        return {
            'theta': theta,
            'scales': torch.stack([s_x, s_y], dim=-1),
            'shear': k,
            'det_Q': det_Q,
        }








class SO3UniformDistribution(td.Distribution):
    arg_constraints = {}
    support = td.constraints.real  # Output is real matrices
    has_rsample = False

    def __init__(
        self,
        validate_args=False,
    ):
        """
        Use Gaussian normalization over quaternions to create uniform SO(3) distribution
        """

        self.group = "SO(3)"
        event_shape = (3, 3)

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=event_shape,
            validate_args=validate_args,
        )

        # Standard normal in R^4 for quaternions before normalization
        self.coeff_dist = td.Independent(td.Normal(torch.zeros(4), torch.ones(4)), 1)

    def to(self, device):
        self.coeff_dist = td.Independent(
            td.Normal(torch.zeros(4, device=device), torch.ones(4, device=device)), 1
        )

    def sample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.batch_shape
        z = self.coeff_dist.sample(shape)  # Unnormalized quaternions [..., 4]
        q = z / z.norm(dim=-1, keepdim=True)  # Normalize to unit quaternions [..., 4]
        return quaternion_to_matrix(q)

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.batch_shape
        z = self.coeff_dist.rsample(shape)  # Unnormalized quaternions [..., 4]
        q = z / z.norm(dim=-1, keepdim=True)  # Normalize to unit quaternions [..., 4]
        return quaternion_to_matrix(q)

    def log_prob(self, value):
        """
        Compute log_prob on SO(3).

        Checks if `value` is a valid rotation matrix:
        - R^T R == I (within tol)
        - det(R) == 1 (within tol)
        If not valid, returns -inf.

        Otherwise, returns constant log probability for uniform distribution on SO(3).
        """

        # Check shape
        if value.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (..., 3, 3) shape, got {value.shape}")

        # Orthogonality check: R^T R ≈ I
        RtR = value.transpose(-2, -1) @ value
        I = torch.eye(3, device=value.device, dtype=value.dtype).expand_as(RtR)
        orthogonality_error = (
            (RtR - I).abs().max(dim=-1)[0].max(dim=-1)[0]
        )  # max abs deviation

        # Determinant check
        det = torch.linalg.det(value)
        det_error = (det - 1).abs()

        tol = 1e-5  # tolerance

        valid_mask = (orthogonality_error < tol) & (det_error < tol)

        # Broadcast mask to batch dims
        valid_mask = valid_mask.reshape(value.shape[:-2])

        log_uniform_density = -math.log(8 * math.pi**2)  # ≈ -6.847

        # Fill -inf where invalid
        out = torch.full(
            value.shape[:-2], float("-inf"), device=value.device, dtype=value.dtype
        )
        out[valid_mask] = log_uniform_density

        return out






class ObjectTransformDistribution(td.Distribution):
    arg_constraints = {}
    support = td.constraints.real
    has_rsample = False

    def __init__(
        self,
        base_dist,
        base_object,
        validate_args=False,
    ):
        """
        Args:
            base_dist: a distribution represented as [K, D, D] transformation matrices
            : the object to transform, shape [..., D]
        """
        self.base_dist = base_dist
        self.base_object = torch.tensor(base_object.data)

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=self.base_object.shape,
            validate_args=validate_args,
        )

    def to(self, device):
        self.base_object = self.base_object.to(device)
        self.base_dist.to(device)

    def sample(self, sample_shape=torch.Size(), return_transform=False):
        """
        Sample R ~ base_dist, return R @ x
        """

        R = self.base_dist.sample(sample_shape)  # [*sample_shape, D, D]
        x = self.base_object  # [N, D] or [...,D]
        
        # Debug: Check R shape
        # R should be [*sample_shape, D, D], e.g., [20000, 6, 6] for sample_shape=(20000,)
        
        # Properly expand x to match sample_shape
        # Convert sample_shape to tuple
        if isinstance(sample_shape, torch.Size):
            sample_shape_tuple = tuple(sample_shape)
        elif isinstance(sample_shape, int):
            sample_shape_tuple = (sample_shape,)
        else:
            sample_shape_tuple = tuple(sample_shape)
        
        # Expand x: [N, D] -> [*sample_shape, N, D]
        # For single batch dimension, use repeat for clarity
        if len(sample_shape_tuple) == 1:
            # [N, D] -> [B, N, D] where B = sample_shape_tuple[0]
            B = sample_shape_tuple[0]
            x = x.unsqueeze(0).repeat(B, 1, 1)  # [1, N, D] -> [B, N, D]
        else:
            # For multiple batch dimensions, use unsqueeze and expand
            x_original_shape = tuple(x.shape)  # (N, D)
            for i in range(len(sample_shape_tuple)):
                x = x.unsqueeze(0)
            expand_shape = sample_shape_tuple + x_original_shape
            x = x.expand(expand_shape)
        
        if R.is_complex():
            x = torch.complex(x, torch.zeros_like(x))

        x = x.type_as(R)

        # Matrix multiplication: x @ R.adjoint()
        # x: [*sample_shape, N, D], R: [*sample_shape, D, D]
        # Need to ensure proper broadcasting
        # For batched matrix multiplication, use bmm when we have a single batch dimension
        if len(sample_shape_tuple) == 1 and len(x.shape) == 3:
            # x: [B, N, D], R should be [B, D, D]
            # Ensure R is 3D for bmm
            if len(R.shape) == 2:
                # R is [D, D], need to expand to [B, D, D]
                B = sample_shape_tuple[0]
                R = R.unsqueeze(0).expand(B, -1, -1)
            elif len(R.shape) == 4:
                # R might be [B, 1, D, D] or [1, B, D, D], squeeze middle dimensions
                # Keep first and last two dimensions
                if R.shape[1] == 1:
                    R = R.squeeze(1)  # [B, 1, D, D] -> [B, D, D]
                elif R.shape[0] == 1:
                    R = R.squeeze(0)  # [1, B, D, D] -> [B, D, D]
                else:
                    # Reshape to [B, D, D]
                    B = sample_shape_tuple[0]
                    D = R.shape[-1]
                    R = R.view(B, D, D)
            elif len(R.shape) != 3:
                # R has unexpected shape, try to reshape it
                # R should be [B, D, D] where B = sample_shape_tuple[0]
                B = sample_shape_tuple[0]
                # Get the last two dimensions (D, D)
                D = R.shape[-1]
                # Reshape to [B, D, D]
                R = R.view(B, D, D)
            # Use batched matrix multiplication
            R_T = R.transpose(-2, -1) if not R.is_complex() else R.adjoint()
            # Ensure R_T is also 3D
            if len(R_T.shape) != 3:
                B = sample_shape_tuple[0]
                D = R_T.shape[-1]
                R_T = R_T.view(B, D, D)
            y = torch.bmm(x, R_T)  # [B, N, D] @ [B, D, D] -> [B, N, D]
        else:
            # For multiple batch dimensions or complex shapes, use @ with proper broadcasting
            # Ensure R has the right shape for broadcasting
            R_T = R.transpose(-2, -1) if not R.is_complex() else R.adjoint()
            y = x @ R_T  # [*sample_shape, N, D]

        if return_transform:
            return y, R
        else:
            return y

    def log_prob(self, value: torch.Tensor):
        """
        Args:
            value: [B, N, D], transformed objects

        Returns:
            log_prob for each batch element, shape [B]
        """
        B = value.shape[0]
        # Flatten spatial dims of value[b] to [B, N, D]
        value_flat = value.reshape(B, -1, self.event_shape[-1])  # [B, N, D]

        # Flatten spatial dims of base_object to [N, D]
        x_flat = self.base_object.reshape(-1, self.event_shape[-1])
        x_flat = x_flat.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        x_flat = x_flat.to(value_flat.device)

        # Solve x @ R^T = value
        R_hat = torch.linalg.lstsq(x_flat, value_flat).solution
        R_hat = R_hat.transpose(-2, -1)  # [B, D, D]

        # Compute log_prob batchwise
        return self.base_dist.log_prob(R_hat)  # [B]









class MultiObjectTransformDistribution(td.Distribution):
    """
    Distribution over transformed objects where the base object is randomly sampled
    from a collection of objects (e.g., ModelNet10 point clouds).
    
    Each sample:
    1. Randomly picks one base object from the collection
    2. Samples a transformation from base_dist
    3. Applies the transformation to the selected object
    
    Args:
        base_dist: Distribution over transformations (e.g., Tet group elements)
        base_objects: Object with .data attribute of shape [num_objects, n_points, D]
                     (e.g., ModelNet10Loader with multiple point clouds)
    """
    arg_constraints = {}
    has_rsample = False

    def __init__(
        self,
        base_dist,
        base_objects,
        validate_args=False,
    ):
        """
        Args:
            base_dist: Distribution over transformations [K, D, D]
            base_objects: Object with .data attribute of shape [num_objects, n_points, D]
        """
        self.base_dist = base_dist
        # base_objects.data shape: [num_objects, n_points, D]
        self.base_objects = torch.tensor(base_objects.data)
        self.num_objects = self.base_objects.shape[0]
        
        # Event shape is the shape of one object: [n_points, D]
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=self.base_objects.shape[1:],  # [n_points, D]
            validate_args=validate_args,
        )

    def to(self, device):
        self.base_objects = self.base_objects.to(device)
        self.base_dist.to(device)

    def sample(self, sample_shape=torch.Size(), return_transform=False):
        """
        Sample transformations and randomly pick base objects, return transformed objects.
        
        Args:
            sample_shape: Shape of samples to generate
            return_transform: Whether to return the transformation matrices
            
        Returns:
            y: Transformed objects, shape [*sample_shape, n_points, D]
            R: (optional) Transformation matrices, shape [*sample_shape, D, D]
        """
        # Sample transformations
        R = self.base_dist.sample(sample_shape)  # [*sample_shape, D, D]
        
        # Convert sample_shape to tuple
        if isinstance(sample_shape, torch.Size):
            sample_shape_tuple = tuple(sample_shape)
        elif isinstance(sample_shape, int):
            sample_shape_tuple = (sample_shape,)
        else:
            sample_shape_tuple = tuple(sample_shape)
        
        # Calculate total number of samples
        if len(sample_shape_tuple) == 0:
            B = 1
        else:
            B = 1
            for s in sample_shape_tuple:
                B *= s
        
        # Randomly select base objects for each sample
        obj_indices = torch.randint(0, self.num_objects, (B,))
        
        # Get selected objects: [B, n_points, D]
        x = self.base_objects[obj_indices]
        
        # Reshape x if needed to match sample_shape
        if len(sample_shape_tuple) > 1:
            x = x.reshape(*sample_shape_tuple, *x.shape[1:])
        
        # Handle complex transformations
        if R.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        
        x = x.type_as(R)
        
        # Matrix multiplication: x @ R.T
        if len(sample_shape_tuple) == 1 and len(x.shape) == 3:
            # x: [B, N, D], R should be [B, D, D]
            if len(R.shape) == 2:
                R = R.unsqueeze(0).expand(B, -1, -1)
            elif len(R.shape) == 4:
                if R.shape[1] == 1:
                    R = R.squeeze(1)
                elif R.shape[0] == 1:
                    R = R.squeeze(0)
                else:
                    D = R.shape[-1]
                    R = R.view(B, D, D)
            elif len(R.shape) != 3:
                D = R.shape[-1]
                R = R.view(B, D, D)
            
            # Use batched matrix multiplication
            R_T = R.transpose(-2, -1) if not R.is_complex() else R.adjoint()
            if len(R_T.shape) != 3:
                D = R_T.shape[-1]
                R_T = R_T.view(B, D, D)
            y = torch.bmm(x, R_T)  # [B, N, D] @ [B, D, D] -> [B, N, D]
        else:
            # For multiple batch dimensions, use @ with broadcasting
            R_T = R.transpose(-2, -1) if not R.is_complex() else R.adjoint()
            y = x @ R_T
        
        if return_transform:
            return y, R
        else:
            return y

    def log_prob(self, value: torch.Tensor):
        """
        Log probability is not well-defined for this distribution.
        Return uniform log probability.
        """
        B = value.shape[0]
        return torch.full((B,), -np.log(self.num_objects), device=value.device)


class MaskedObjectTransformDistribution(td.Distribution):
    """
    Distribution over transformed objects with random point masking.
    
    This wraps MultiObjectTransformDistribution and adds random masking:
    1. Sample from base distribution (object + transformation)
    2. Randomly select n_masked_points and set them to mask_value
    
    Args:
        base_dist: Distribution over transformations (e.g., Tet group elements)
        base_objects: Object with .data attribute of shape [num_objects, n_points, D]
        n_masked_points: Number of points to randomly mask (default: 10)
        mask_value: Value to set masked points to (default: 0.0)
        random_seed: Random seed for reproducibility (default: None, means random)
    """
    arg_constraints = {}
    has_rsample = False

    def __init__(
        self,
        base_dist,
        base_objects,
        n_masked_points=10,
        mask_value=0.0,
        random_seed=None,
        validate_args=False,
    ):
        """
        Args:
            base_dist: Distribution over transformations [K, D, D]
            base_objects: Object with .data attribute of shape [num_objects, n_points, D]
            n_masked_points: Number of points to randomly mask
            mask_value: Value to set masked points to
            random_seed: Random seed for reproducibility
        """
        # Store base_dist for compatibility with other code that accesses it
        self.base_dist = base_dist
        
        # Use MultiObjectTransformDistribution as the base
        self.base_multi_dist = MultiObjectTransformDistribution(
            base_dist=base_dist,
            base_objects=base_objects,
            validate_args=validate_args,
        )
        
        # Expose some attributes from base_multi_dist for compatibility
        self.base_objects = self.base_multi_dist.base_objects
        self.num_objects = self.base_multi_dist.num_objects
        
        self.n_masked_points = n_masked_points
        self.mask_value = mask_value
        self.random_seed = random_seed
        
        # Event shape is the same as base distribution: [n_points, D]
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=self.base_multi_dist.event_shape,
            validate_args=validate_args,
        )

    def to(self, device):
        self.base_multi_dist.to(device)
        return self

    def sample(self, sample_shape=torch.Size(), return_transform=False, return_mask=False):
        """
        Sample transformed objects and apply random masking.
        
        Args:
            sample_shape: Shape of samples to generate
            return_transform: Whether to return the transformation matrices
            return_mask: Whether to return the mask indicating which points were masked
            
        Returns:
            y: Masked transformed objects, shape [*sample_shape, n_points, D]
            R: (optional) Transformation matrices, shape [*sample_shape, D, D]
            mask: (optional) Boolean mask, shape [*sample_shape, n_points] (True = masked)
        """
        # Sample from base distribution
        if return_transform:
            y, R = self.base_multi_dist.sample(sample_shape, return_transform=True)
        else:
            y = self.base_multi_dist.sample(sample_shape, return_transform=False)
        
        # Convert sample_shape to tuple
        if isinstance(sample_shape, torch.Size):
            sample_shape_tuple = tuple(sample_shape)
        elif isinstance(sample_shape, int):
            sample_shape_tuple = (sample_shape,)
        else:
            sample_shape_tuple = tuple(sample_shape)
        
        # Calculate batch size
        if len(sample_shape_tuple) == 0:
            B = 1
            y = y.unsqueeze(0)  # Add batch dimension
        else:
            B = 1
            for s in sample_shape_tuple:
                B *= s
        
        # y shape: [B, n_points, D]
        n_points = y.shape[-2]
        D = y.shape[-1]
        
        # Reshape y to [B, n_points, D] for easier processing
        y_flat = y.reshape(B, n_points, D)
        
        # Create mask for each sample
        mask = torch.zeros(B, n_points, dtype=torch.bool, device=y.device)
        
        # Use numpy for random sampling with seed control
        if self.random_seed is not None:
            rng = np.random.RandomState(self.random_seed)
        else:
            rng = np.random
        
        # Randomly select points to mask for each sample
        for i in range(B):
            if self.n_masked_points > 0 and self.n_masked_points < n_points:
                # Randomly select indices to mask
                masked_indices = rng.choice(n_points, size=self.n_masked_points, replace=False)
                mask[i, masked_indices] = True
                # Set masked points to mask_value
                y_flat[i, masked_indices, :] = self.mask_value
        
        # Reshape back to original shape
        if len(sample_shape_tuple) == 0:
            y_flat = y_flat.squeeze(0)
        else:
            y_flat = y_flat.reshape(*sample_shape_tuple, n_points, D)
        
        # Return based on flags
        if return_transform and return_mask:
            if len(sample_shape_tuple) == 0:
                mask = mask.squeeze(0)
            else:
                mask = mask.reshape(*sample_shape_tuple, n_points)
            return y_flat, R, mask
        elif return_transform:
            return y_flat, R
        elif return_mask:
            if len(sample_shape_tuple) == 0:
                mask = mask.squeeze(0)
            else:
                mask = mask.reshape(*sample_shape_tuple, n_points)
            return y_flat, mask
        else:
            return y_flat

    def log_prob(self, value: torch.Tensor):
        """
        Log probability delegates to base distribution.
        """
        return self.base_multi_dist.log_prob(value)




class NoisyMaskedObjectTransformDistribution(td.Distribution):
    """
    Distribution over transformed objects with random masking and SO(3) noise.

    Sampling pipeline:
      1. Pick a base object from the collection.
      2. Sample a discrete group element g from base_dist (e.g. Icosahedral group).
      3. Apply g to the object: y = x @ g^T.
      4. Randomly zero out n_masked_points points.
      5. Sample a small SO(3) noise rotation R_noise near the identity and apply it:
         y_noisy = y @ R_noise^T.

    The noise is generated via the exponential map:
      omega ~ N(0, noise_scale^2 * I_3),
      R_noise = expm(skew(omega))
    where noise_scale (in radians) controls how far from the identity the noise is.
    Setting noise_scale=0 disables the noise entirely.

    Args:
        base_dist:        Distribution over group transformations (e.g. DiscreteDeltaMixture).
        base_objects:     Object with a .data attribute of shape [num_objects, n_points, D].
        n_masked_points:  Number of points to randomly zero out (default: 0).
        mask_value:       Replacement value for masked points (default: 0.0).
        random_seed:      Seed for the masking RNG; None means a new seed each call.
        noise_scale:      Standard deviation (radians) of the rotation-angle noise (default: 0.1).
    """

    arg_constraints = {}
    has_rsample = False

    def __init__(
        self,
        base_dist,
        base_objects,
        n_masked_points: int = 0,
        mask_value: float = 0.0,
        random_seed=None,
        noise_scale: float = 0.1,
        validate_args=False,
    ):
        self.noise_scale = noise_scale

        # Reuse MaskedObjectTransformDistribution for steps 1-4
        self.masked_dist = MaskedObjectTransformDistribution(
            base_dist=base_dist,
            base_objects=base_objects,
            n_masked_points=n_masked_points,
            mask_value=mask_value,
            random_seed=random_seed,
            validate_args=validate_args,
        )

        # Expose attributes expected by downstream code
        self.base_dist = base_dist
        self.base_objects = self.masked_dist.base_objects
        self.num_objects = self.masked_dist.num_objects

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=self.masked_dist.event_shape,
            validate_args=validate_args,
        )

    def to(self, device):
        self.masked_dist.to(device)
        return self

    def _sample_so3_noise(self, batch_size: int, device) -> torch.Tensor:
        """
        Sample batch_size SO(3) rotation matrices close to the identity.

        Uses the Lie-algebra exponential map:
          omega ~ N(0, noise_scale^2 * I_3)  (rotation vector, radians)
          R = expm(skew(omega))

        Returns: [batch_size, 3, 3]
        """
        if self.noise_scale == 0.0:
            return torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        omega = torch.randn(batch_size, 3, device=device) * self.noise_scale  # [B, 3]

        # Build skew-symmetric matrices:  skew(w) = [[0, -w2, w1], [w2, 0, -w0], [-w1, w0, 0]]
        zeros = torch.zeros(batch_size, device=device)
        w0, w1, w2 = omega[:, 0], omega[:, 1], omega[:, 2]

        skew = torch.stack(
            [
                torch.stack([zeros,  -w2,   w1], dim=-1),
                torch.stack([w2,   zeros,  -w0], dim=-1),
                torch.stack([-w1,    w0, zeros], dim=-1),
            ],
            dim=-2,
        )  # [B, 3, 3]

        return torch.linalg.matrix_exp(skew)  # [B, 3, 3]

    def sample(
        self,
        sample_shape=torch.Size(),
        return_transform: bool = False,
        return_mask: bool = False,
    ):
        """
        Sample noisy transformed point clouds.

        Args:
            sample_shape:      Batch shape for sampling.
            return_transform:  If True, also return the discrete group element R.
            return_mask:       If True, also return the boolean mask of masked points.

        Returns:
            y_noisy:  Transformed + masked + noise-perturbed point cloud,
                      shape [*sample_shape, n_points, D].
            R:        (optional) Discrete group rotation, shape [*sample_shape, D, D].
            mask:     (optional) Boolean mask, shape [*sample_shape, n_points].
        """
        # --- Step 1-4: sample from the masked distribution ---
        if return_mask:
            out = self.masked_dist.sample(
                sample_shape, return_transform=return_transform, return_mask=True
            )
            if return_transform:
                y, R, mask = out
            else:
                y, mask = out
        else:
            out = self.masked_dist.sample(
                sample_shape, return_transform=return_transform, return_mask=False
            )
            if return_transform:
                y, R = out
            else:
                y = out

        # --- Normalise sample_shape ---
        if isinstance(sample_shape, torch.Size):
            shape_tuple = tuple(sample_shape)
        elif isinstance(sample_shape, int):
            shape_tuple = (sample_shape,)
        else:
            shape_tuple = tuple(sample_shape)

        scalar_sample = len(shape_tuple) == 0
        if scalar_sample:
            y = y.unsqueeze(0)

        B = y.shape[0] if not scalar_sample else 1
        for s in shape_tuple[1:]:
            B *= s

        n_points = y.shape[-2]
        D = y.shape[-1]
        y_flat = y.reshape(B, n_points, D)

        # --- Step 5: apply SO(3) noise ---
        R_noise = self._sample_so3_noise(B, device=y.device)        # [B, 3, 3]
        y_noisy = torch.bmm(y_flat, R_noise.transpose(-2, -1))       # [B, N, 3]

        # Restore original shape
        if scalar_sample:
            y_noisy = y_noisy.squeeze(0)
        else:
            y_noisy = y_noisy.reshape(*shape_tuple, n_points, D)

        # --- Assemble return tuple ---
        if return_transform and return_mask:
            return y_noisy, R, mask
        elif return_transform:
            return y_noisy, R
        elif return_mask:
            return y_noisy, mask
        else:
            return y_noisy

    def log_prob(self, value: torch.Tensor):
        """Delegates to the underlying masked distribution."""
        return self.masked_dist.log_prob(value)




