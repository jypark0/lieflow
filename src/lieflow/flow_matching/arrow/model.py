from collections import defaultdict
from functools import reduce

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lieflow.distributions import (
    DiscreteDeltaMixture,
    RandomDiscreteDeltaMixture,
)
from lieflow.utils import (
    batched_object_div,
    blogm,
    bracket_so3,
    exp_so3,
    expand_like,
    icdf_beta,
    icdf_power,
    log_SO3,
    sample_beta_distribution,
    sample_power_distribution,
)

# helpers functions
expm = torch.linalg.matrix_exp


# Model
class Flow(nn.Module):
    def __init__(
        self,
        net,
        prior_dist,
        device,
        max_grad_norm=None,
        time_sampling="uniform",
        time_sampling_kwargs={},
        lambda_reg=0.0,
    ):
        super().__init__()
        self.device = device
        self.prior_dist = prior_dist
        self.prior_dist.to(device)
        self.lambda_reg = lambda_reg
        self.net = net

        self.max_grad_norm = max_grad_norm
        self.time_sampling = time_sampling
        if time_sampling == "power":
            self.time_skewness = time_sampling_kwargs["skewness"]
        elif time_sampling == "beta":
            self.time_alpha = time_sampling_kwargs["alpha"]
            self.time_beta = time_sampling_kwargs["beta"]

    @torch.no_grad()
    def sample_all(self, x_1, n_steps, return_transform=False, x_0=None):
        if x_0 is None:
            B = x_1.shape[0]
            rand_transform = self.prior_dist.sample((B,)).to(self.device)

            x_t = x_1 @ rand_transform.transpose(-2, -1)
        else:
            B = x_0.shape[0]
            x_t = x_0.clone().to(self.device)

        x_t_all = [x_t]
        transforms = torch.zeros(n_steps + 1, B, *self.prior_dist.event_shape, device=self.device)
        if self.time_sampling == "uniform":
            # Evenly spaced timesteps
            t_values = torch.linspace(0.0, 1.0, n_steps + 1, device=self.device)
        elif self.time_sampling == "power":
            t_values = icdf_power(
                0,
                1,
                n_steps + 1,
                skewness=self.time_skewness,
                device=self.device,
            )
        elif self.time_sampling == "beta":
            t_values = icdf_beta(
                self.time_alpha,
                self.time_beta,
                n_steps + 1,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown time sampling: {self.time_sampling}")

        for i in range(n_steps):
            t = t_values[i]
            delta_t = t_values[i + 1] - t_values[i]

            A = self.net(x_t, t)
            tf = expm(delta_t * A)
            x_t = x_t @ tf

            x_t_all.append(x_t)
            transforms[i] = tf

        x_t_all = torch.stack(x_t_all, dim=0)  # [n_steps + 1, batch, 2, 2]
        # Last transform is identity
        transforms[-1] = torch.eye(x_1.shape[-1], device=self.device).unsqueeze(0).expand(B, -1, -1)

        if return_transform:
            if x_0 is None:
                return x_t_all, rand_transform, transforms
            else:
                return x_t_all, transforms
        else:
            return x_t_all

    @torch.no_grad()
    def sample(self, x_1, n_steps, return_transform=False, x_0=None):
        if return_transform:
            if x_0 is None:
                x_t_all, orig_tf, tfs = self.sample_all(x_1, n_steps, return_transform=return_transform, x_0=x_0)
                return x_t_all[-1], orig_tf, reduce(torch.bmm, tfs)
            else:
                x_t_all, tfs = self.sample_all(x_1, n_steps, return_transform=return_transform, x_0=x_0)
                return x_t_all[-1], reduce(torch.bmm, tfs)
        else:
            x_t_all = self.sample_all(x_1, n_steps, return_transform=return_transform, x_0=x_0)
            return x_t_all[-1]

    def interpolate(self, x_0: Tensor, x_1: Tensor, t, random_transform=None) -> Tensor:

        solv = torch.linalg.lstsq(x_0, x_1, driver="gels").solution
        tf = blogm(solv)

        x_t = x_0 @ expm(expand_like(t, tf) * tf)

        return x_t, tf

    def interpolate_with_noise(self, x_0: Tensor, x_1: Tensor, t) -> Tensor:
        # x_0 @ expm(tf^T) = x_1, tf = blogm(x_0^{-1} @ x_1)^T
        # Let tf be the target so the model learns to predict
        # the transpose of transformation matrices
        tf = blogm(torch.linalg.lstsq(x_0, x_1, driver="gels").solution)

        # Add noise to tf
        noise = torch.randn_like(tf) * 0.1  # Adjust noise scale as needed
        tf += noise

        x_t = x_0 @ expm(expand_like(t, tf) * tf)

        return x_t, tf

    def train_net(self, train_loader, optimizer):
        self.train()
        results = defaultdict(float)
        for x_1 in train_loader:
            B = x_1.shape[0]

            x_1 = x_1.to(self.device)
            if self.time_sampling == "uniform":
                # Uniformly sample t in [0, 1]
                t = torch.rand((B,), device=self.device)
            elif self.time_sampling == "power":
                t = sample_power_distribution(0, 1, skewness=self.time_skewness, size=(B,), device=self.device)
            elif self.time_sampling == "beta":
                t = sample_beta_distribution(
                    self.time_alpha,
                    self.time_beta,
                    B,
                    device=self.device,
                )
            else:
                raise ValueError(f"Unknown time sampling: {self.time_sampling}")

            # sample_base
            rand_transforms = self.prior_dist.sample((B,)).to(self.device)
            x_0 = x_1 @ rand_transforms.transpose(-2, -1)

            x_t, target = self.interpolate(x_0, x_1, t, random_transform=rand_transforms)

            optimizer.zero_grad()
            pred = self.net(x_t, t)

            mse_loss = F.mse_loss(pred, target)
            results["mse_loss"] += mse_loss.item()

            # Regularization to prevent small rotations
            # Encourage the model to predict non-zero velocities
            pred_norm = torch.norm(pred.reshape(B, -1), dim=1)
            target_norm = torch.norm(target.reshape(B, -1), dim=1)

            # Penalize when predicted norm is much smaller than target norm
            norm_ratio = pred_norm / (target_norm + 1e-6)
            reg_loss = torch.mean((1 - norm_ratio) ** 2)

            results["reg_loss"] += reg_loss.item()

            # Total loss
            loss = mse_loss + self.lambda_reg * reg_loss
            results["loss"] += loss.item()

            loss.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)

            optimizer.step()

        for k in results.keys():
            results[k] /= len(train_loader)

        return results

    @torch.no_grad()
    def eval_net(self, test_loader):
        self.eval()
        results = defaultdict(float)

        for x_1, gt_transform in test_loader:
            x_1 = x_1.to(self.device)
            gt_transform = gt_transform.to(self.device)

            B = x_1.shape[0]
            t = torch.rand(B, device=self.device)

            rand_transforms = self.prior_dist.sample((B,)).to(self.device)
            x_0 = x_1 @ rand_transforms.transpose(-2, -1)

            x_t, target = self.interpolate(x_0, x_1, t, random_transform=rand_transforms)

            pred = self.net(x_t, t)

            mse_loss = F.mse_loss(pred, target)
            results["mse_loss"] += mse_loss.item()

        for k in results.keys():
            results[k] /= len(test_loader)

        return results

    def _matrix_geodesic_inverse(self, x_t: Tensor, x_1: Tensor, t: float) -> Tensor:
        """
        Given x_t and x_1, find x_0.

        x_1 = x_0 @ expm(A)
        Let A = logm(x_0^{-1} @ x_1).

        Derivation:
            x_t^{-1} @ x_1 = (x_0 @ expm(t A))^{-1} @ (x_0 @ expm(A))
            expm(-t A) x_0^{-1} x_0 expm(A) = expm((1-t)A)
            A = \frac{1}{1-t} logm(x_t^{-1} @ x_1)
            x_0 = x_t @ expm(t/(t-1) * logm(x_t^{-1} @ x_1))
        """
        if abs(t - 1.0) < 1e-6:
            # At t=1, x_t = x_1, so any x_0 on the geodesic works
            # Return x_t as a reasonable choice
            return x_t

        # Ensure inputs are on the manifold
        # x_t = self.prior_dist.project_to_manifold(x_t)
        # x_1 = self.prior_dist.project_to_manifold(x_1)

        # x_0 = x_t @ expm((t/(t-1)) * logm(x_t^{-1} @ x_1))
        x_t_inv_x_1 = torch.linalg.lstsq(x_t, x_1, driver="gels").solution
        log_term = blogm(x_t_inv_x_1)
        x_0 = x_t @ expm((t / (t - 1.0)) * log_term)

        # Ensure x_0 is on manifold
        # x_0 = self.prior_dist.project_to_manifold(x_0)

        return x_0

    def _forward_simulate(self, x_0: Tensor, t_values: Tensor) -> tuple[Tensor, Tensor]:
        """Forward simulate from x_0 to get trajectories and log probabilities."""
        batch_size = x_0.shape[0]

        # Ensure x_0 is on manifold
        # x_0 = self.prior_dist.project_to_manifold(x_0)

        x_t = x_0
        x_t_all = [x_t]

        log_p_t = self.prior_dist.log_prob(x_0)
        log_p_all = [log_p_t]

        for i in range(len(t_values) - 1):
            t = t_values[i]
            delta_t = t_values[i + 1] - t_values[i]
            t_tensor = torch.full((batch_size, 1), t.item(), device=self.device)

            # Compute divergence
            with torch.set_grad_enabled(True):
                x_t_grad = x_t.detach().requires_grad_(True)
                div = batched_object_div(self.net.forward, x_t_grad, t_tensor)

            # Exponential Euler step
            A = self.net(x_t, t_tensor)
            x_t = x_t @ expm(delta_t * A)

            # Project back to manifold
            # x_t = self.prior_dist.project_to_manifold(x_t)

            log_p_t = log_p_t - delta_t * div

            x_t_all.append(x_t)
            log_p_all.append(log_p_t)

        x_t_all = torch.stack(x_t_all, dim=0)  # [n_steps + 1, batch, 2, 2]
        log_p_all = torch.stack(log_p_all, dim=0)

        return x_t_all, log_p_all

    def _integrate_divergence_to_time(self, x_0: Tensor, t: float, n_substeps: int) -> Tensor:
        """Integrate divergence from x_0 to time t."""
        batch_size = x_0.shape[0]
        delta_t = t / n_substeps

        x_t = x_0
        log_p_t = self.prior_dist.log_prob(x_0)  # [batch, M]

        for i in range(n_substeps):
            s = i * delta_t
            s_tensor = torch.full((batch_size, 1), s, device=self.device)

            # Compute divergence
            with torch.set_grad_enabled(True):
                x_t_grad = x_t.detach().requires_grad_(True)
                div = batched_object_div(self.net.forward, x_t_grad, s_tensor)

            # Exponential Euler step
            A = self.net(x_t, s_tensor)
            x_t = x_t @ expm(delta_t * A)
            log_p_t = log_p_t - delta_t * div

        # Return log probability at time t
        return log_p_t

    @torch.no_grad()
    def approx_p_x1_given_xt(
        self,
        x_0: torch.Tensor,
        p1_dist,  # a torch.distributions instance
        n_steps: int = 50,
        n_x1_grid: int = 100,
    ) -> dict[str, torch.Tensor]:
        """
        Compute p(x1 | x_t) for flow matching on prior group.
        """
        self.eval()
        device = x_0.device
        batch_size = x_0.shape[0]

        # Time grid
        if self.time_sampling == "uniform":
            # Evenly spaced timesteps
            t_values = torch.linspace(0.0, 1.0, n_steps + 1, device=self.device)
        elif self.time_sampling == "power":
            t_values = icdf_power(
                0,
                1,
                n_steps + 1,
                skewness=self.time_skewness,
                device=self.device,
            )
        elif self.time_sampling == "beta":
            t_values = icdf_beta(
                self.time_alpha,
                self.time_beta,
                n_steps + 1,
                device=self.device,
            )

        # Detect discrete target
        use_discrete = isinstance(p1_dist, DiscreteDeltaMixture) or isinstance(p1_dist, RandomDiscreteDeltaMixture)

        if use_discrete:
            # Discrete atoms: use matrix means directly
            base_obj = p1_dist.base_object.to(device)
            base_tfs = p1_dist.locs.to(device)

            x1_values = torch.einsum("mdd,nd->mnd", base_tfs, base_obj)  # [M, N, D]

            prior_weights = p1_dist.weights.to(device)
            log_prior_weights = torch.log(prior_weights)
            M = x1_values.shape[0]
        else:
            raise NotImplementedError()
            # # For continuous prior, sample from this distribution
            # x1_values = p1_dist.sample((n_x1_grid,)).to(device)  # [M, D, D]
            # M = x1_values.shape[0]
            #
            # log_p_x1_grid = p1_dist.log_prob(x1_values).view(M)

        event_shape = self.prior_dist.event_shape
        x_shape = x_0.shape[1:]

        # Storage tensors
        x_t_all = torch.zeros(n_steps + 1, batch_size, *x_shape, device=device)
        log_p_t_all = torch.zeros(n_steps + 1, batch_size, device=device)
        p_x1_given_xt_all = torch.zeros(n_steps + 1, batch_size, M, device=device)
        # For GMM, we also track component weights
        if not use_discrete and hasattr(p1_dist, "mixture_distribution"):
            K = p1_dist.mixture_distribution.probs.shape[0]
            weights_all = torch.zeros(n_steps + 1, batch_size, K, device=device)
        else:
            weights_all = torch.zeros(n_steps + 1, batch_size, M, device=device)

        # Forward simulate once for trajectories and marginal log p_t
        x_sim, logp_sim = self._forward_simulate(x_0, t_values)
        x_t_all[:] = x_sim
        log_p_t_all[:] = logp_sim

        # Main loop over times
        for i, t in enumerate(t_values):
            x_t = x_t_all[i]  # [batch, D, D]
            t_val = t.item()

            if t_val == 0.0:
                # At t=0, posterior equals prior
                if use_discrete:
                    post = prior_weights.unsqueeze(0).expand(batch_size, -1)  # [batch, K]
                    p_x1_given_xt_all[i] = post
                    weights_all[i] = post
                else:
                    p_x1 = torch.exp(log_p_x1_grid)
                    p_x1 = p_x1 / p_x1.sum()
                    p_x1_given_xt_all[i] = p_x1.unsqueeze(0).expand(batch_size, -1)

                    # Compute GMM component weights if applicable
                    if hasattr(p1_dist, "mixture_distribution"):
                        weights_all[i] = self._compute_gmm_weights_matrix(x1_values, p_x1.unsqueeze(0), p1_dist).expand(
                            batch_size, -1
                        )
                continue

            # For t > 0, compute likelihoods
            log_lik = torch.full((batch_size, M), -float("inf"), device=device)

            if t_val < 0.99:
                # Matrix geodesic inversion
                batch_M = batch_size * M
                x_t_exp = x_t.unsqueeze(1).expand(-1, M, -1, -1)  # [batch, M, N, D]
                x1_exp = x1_values.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch, M, N, D]

                x_t_flat = x_t_exp.reshape(batch_M, *x_shape)

                x1_flat = x1_exp.reshape(batch_M, *x_shape)

                # Compute x_0 for all pairs
                x0_all = self._matrix_geodesic_inverse(x_t_flat, x1_flat, t_val)

                # Compute log probability by integrating divergence
                log_p = self._integrate_divergence_to_time(x0_all, t_val, n_substeps=i)
                log_lik = log_p.view(batch_size, M)
            else:
                # Near t=1: use geodesic distance on prior group
                x_t_exp = x_t.unsqueeze(1).expand(-1, M, -1, -1)  # [batch, M, N, D]
                x1_exp = x1_values.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch, M, N, D]

                # Geodesic distance: ||log(x_t^{-1} @ x_1)||_F
                x_t_inv_x1 = torch.linalg.lstsq(
                    x_t_exp.reshape(batch_M, *x_shape),
                    x1_exp.reshape(batch_M, *x_shape),
                ).solution

                log_matrices = blogm(x_t_inv_x1).view(batch_size, M, event_shape[-1], event_shape[-1])
                distances = torch.norm(log_matrices.view(batch_size, M, -1), dim=-1)  # [batch, M]

                sigma = 0.1
                log_lik = -0.5 * (distances / sigma) ** 2

            # Combine with prior
            if use_discrete:
                log_post_unnorm = log_lik + log_prior_weights.unsqueeze(0)
            else:
                log_post_unnorm = log_lik + log_p_x1_grid.unsqueeze(0)

            # Normalize
            all_inf_mask = ~torch.isfinite(log_post_unnorm).any(dim=1)
            log_post_unnorm[all_inf_mask] = 0.0
            post = F.softmax(log_post_unnorm, dim=1)
            p_x1_given_xt_all[i] = post

            # Update weights
            if use_discrete:
                weights_all[i] = post
            elif hasattr(p1_dist, "mixture_distribution"):
                weights_all[i] = self._compute_gmm_weights_matrix(x1_values, post, p1_dist)

        # Extract component means as matrices
        if use_discrete:
            component_means = x1_values.clone()
        elif hasattr(p1_dist, "component_distribution"):
            # Convert GMM component means to matrices
            component_means = p1_dist.component_distribution.loc
        else:
            component_means = None

        return {
            "x_t_all": x_t_all,
            "log_p_t_all": log_p_t_all,
            "p_x1_given_xt_all": p_x1_given_xt_all,
            "weights_all": weights_all,
            "x1_grid": x1_values,
            "t_values": t_values,
            "component_means": component_means,  # [K, 2, 2]
            "use_discrete": use_discrete,
        }

    def _compute_gmm_weights_matrix(self, x1_matrices: Tensor, p_x1: Tensor, p1_dist) -> Tensor:
        """
        Compute posterior mixture weights for GMM components in GL+(2) space.

        Args:
            x1_matrices: [M, 2, 2] sampled matrices
            p_x1: [batch, M] posterior probabilities
            p1_dist: GMM distribution in GL+(2) space

        Returns:
            weights: [batch, K] component weights
        """
        if not hasattr(p1_dist, "mixture_distribution"):
            # Not a GMM, return p_x1 as weights
            return p_x1

        mix_probs = p1_dist.mixture_distribution.probs  # [K]
        K = mix_probs.shape[0]
        M = x1_matrices.shape[0]

        # Evaluate each component at each grid point
        log_mix = torch.log(mix_probs)  # [K]

        # For each component k, evaluate log p_k(x1_matrices)
        log_q = torch.zeros(K, M, device=x1_matrices.device)  # [K, M]

        for k in range(K):
            # Get k-th component distribution
            comp_k = p1_dist.component_distribution
            # Evaluate at all grid points
            log_q[k] = comp_k.log_prob(x1_matrices)  # [M]

        # Evaluate total log p1(x1_matrices)
        log_p1 = p1_dist.log_prob(x1_matrices)  # [M]

        # Compute ratio: p(k) * p(x|k) / p(x)
        log_ratio = log_mix.unsqueeze(1) + log_q - log_p1.unsqueeze(0)  # [K, M]
        ratio = torch.exp(log_ratio)

        # Compute weights: sum over grid points weighted by posterior
        weights_unnorm = p_x1 @ ratio.t()  # [batch, K]
        weights = weights_unnorm / (weights_unnorm.sum(dim=1, keepdim=True) + 1e-12)

        return weights


# Model
class Flow_SO3(nn.Module):
    def __init__(
        self,
        net,
        prior_dist,
        device,
        max_grad_norm=None,
        time_sampling="uniform",
        time_sampling_kwargs={},
        lambda_reg=0.0,
    ):
        super().__init__()
        self.device = device
        self.prior_dist = prior_dist
        self.prior_dist.to(device)
        self.lambda_reg = lambda_reg
        self.net = net

        self.max_grad_norm = max_grad_norm
        self.time_sampling = time_sampling
        if time_sampling == "power":
            self.time_skewness = time_sampling_kwargs["skewness"]
        elif time_sampling == "beta":
            self.time_alpha = time_sampling_kwargs["alpha"]
            self.time_beta = time_sampling_kwargs["beta"]

    @torch.no_grad()
    def sample_all(self, x_1, n_steps, return_transform=False, x_0=None):
        if x_0 is None:
            B = x_1.shape[0]
            rand_transform = self.prior_dist.sample((B,)).to(self.device)

            x_t = x_1 @ rand_transform.transpose(-2, -1)
        else:
            B = x_0.shape[0]
            x_t = x_0.clone().to(self.device)

        x_t_all = [x_t]
        transforms = torch.zeros(n_steps + 1, B, *self.prior_dist.event_shape, device=self.device)
        if self.time_sampling == "uniform":
            # Evenly spaced timesteps
            t_values = torch.linspace(0.0, 1.0, n_steps + 1, device=self.device)
        elif self.time_sampling == "power":
            t_values = icdf_power(
                0,
                1,
                n_steps + 1,
                skewness=self.time_skewness,
                device=self.device,
            )
        elif self.time_sampling == "beta":
            t_values = icdf_beta(
                self.time_alpha,
                self.time_beta,
                n_steps + 1,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown time sampling: {self.time_sampling}")

        for i in range(n_steps):
            t = t_values[i]
            delta_t = t_values[i + 1] - t_values[i]

            A_vec = self.net(x_t, t)
            A = bracket_so3(A_vec)
            tf = exp_so3(delta_t * A)
            x_t = x_t @ tf

            x_t_all.append(x_t)
            transforms[i] = tf

        x_t_all = torch.stack(x_t_all, dim=0)  # [n_steps + 1, batch, 2, 2]
        # Last transform is identity
        transforms[-1] = torch.eye(x_1.shape[-1], device=self.device).unsqueeze(0).expand(B, -1, -1)

        if return_transform:
            if x_0 is None:
                return x_t_all, rand_transform, transforms
            else:
                return x_t_all, transforms
        else:
            return x_t_all

    @torch.no_grad()
    def sample(self, x_1, n_steps, return_transform=False, x_0=None):
        if return_transform:
            if x_0 is None:
                x_t_all, orig_tf, tfs = self.sample_all(x_1, n_steps, return_transform=return_transform, x_0=x_0)
                return x_t_all[-1], orig_tf, reduce(torch.bmm, tfs)
            else:
                x_t_all, tfs = self.sample_all(x_1, n_steps, return_transform=return_transform, x_0=x_0)
                return x_t_all[-1], reduce(torch.bmm, tfs)
        else:
            x_t_all = self.sample_all(x_1, n_steps, return_transform=return_transform, x_0=x_0)
            return x_t_all[-1]

    def interpolate(self, x_0: Tensor, x_1: Tensor, t) -> Tensor:
        # x_0 @ expm(tf^T) = x_1, tf = blogm(x_0^{-1} @ x_1)^T
        # Let tf be the target so the model learns to predict
        # the transpose of transformation matrices
        # tf = blogm(torch.linalg.lstsq(x_0, x_1, driver="gels").solution)
        solv = torch.linalg.lstsq(x_0, x_1, driver="gels").solution
        tf = log_SO3(solv)

        x_t = x_0 @ exp_so3(expand_like(t, tf) * tf)

        return x_t, tf

    def interpolate_with_noise(self, x_0: Tensor, x_1: Tensor, t) -> Tensor:
        # x_0 @ expm(tf^T) = x_1, tf = blogm(x_0^{-1} @ x_1)^T
        # Let tf be the target so the model learns to predict
        # the transpose of transformation matrices
        tf = blogm(torch.linalg.lstsq(x_0, x_1, driver="gels").solution)

        # Add noise to tf
        noise = torch.randn_like(tf) * 0.1  # Adjust noise scale as needed
        tf += noise

        x_t = x_0 @ expm(expand_like(t, tf) * tf)

        return x_t, tf

    def train_net(self, train_loader, optimizer):
        self.train()
        results = defaultdict(float)
        for x_1 in train_loader:
            B = x_1.shape[0]

            x_1 = x_1.to(self.device)
            if self.time_sampling == "uniform":
                # Uniformly sample t in [0, 1]
                t = torch.rand((B,), device=self.device)
            elif self.time_sampling == "power":
                t = sample_power_distribution(0, 1, skewness=self.time_skewness, size=(B,), device=self.device)
            elif self.time_sampling == "beta":
                t = sample_beta_distribution(
                    self.time_alpha,
                    self.time_beta,
                    B,
                    device=self.device,
                )
            else:
                raise ValueError(f"Unknown time sampling: {self.time_sampling}")

            # sample_base
            rand_transforms = self.prior_dist.sample((B,)).to(self.device)
            x_0 = x_1 @ rand_transforms.transpose(-2, -1)

            x_t, target = self.interpolate(x_0, x_1, t)
            target = bracket_so3(target)
            optimizer.zero_grad()
            pred = self.net(x_t, t)

            mse_loss = F.mse_loss(pred, target)
            results["mse_loss"] += mse_loss.item()

            # Regularization to prevent small rotations
            # Encourage the model to predict non-zero velocities
            pred_norm = torch.norm(pred.reshape(B, -1), dim=1)
            target_norm = torch.norm(target.reshape(B, -1), dim=1)

            # Penalize when predicted norm is much smaller than target norm
            norm_ratio = pred_norm / (target_norm + 1e-6)
            reg_loss = torch.mean((1 - norm_ratio) ** 2)

            results["reg_loss"] += reg_loss.item()

            # Total loss
            loss = mse_loss + self.lambda_reg * reg_loss
            results["loss"] += loss.item()

            loss.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)

            optimizer.step()

        for k in results.keys():
            results[k] /= len(train_loader)

        return results

    @torch.no_grad()
    def eval_net(self, test_loader):
        self.eval()
        results = defaultdict(float)

        for batch in test_loader:
            # Handle both cases: with or without gt_transform
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_1, gt_transform = batch
            else:
                x_1 = batch
                gt_transform = None  # Not used in evaluation anyway

            x_1 = x_1.to(self.device)
            if gt_transform is not None:
                gt_transform = gt_transform.to(self.device)

            B = x_1.shape[0]
            t = torch.rand(B, device=self.device)

            rand_transforms = self.prior_dist.sample((B,)).to(self.device)
            x_0 = x_1 @ rand_transforms.transpose(-2, -1)

            x_t, target = self.interpolate(x_0, x_1, t)
            target = bracket_so3(target)

            pred = self.net(x_t, t)

            mse_loss = F.mse_loss(pred, target)
            results["mse_loss"] += mse_loss.item()

        for k in results.keys():
            results[k] /= len(test_loader)

        return results

    def _matrix_geodesic_inverse(self, x_t: Tensor, x_1: Tensor, t: float) -> Tensor:
        """
        Given x_t and x_1, find x_0.

        x_1 = x_0 @ expm(A)
        Let A = logm(x_0^{-1} @ x_1).

        Derivation:
            x_t^{-1} @ x_1 = (x_0 @ expm(t A))^{-1} @ (x_0 @ expm(A))
            expm(-t A) x_0^{-1} x_0 expm(A) = expm((1-t)A)
            A = \frac{1}{1-t} logm(x_t^{-1} @ x_1)
            x_0 = x_t @ expm(t/(t-1) * logm(x_t^{-1} @ x_1))
        """
        if abs(t - 1.0) < 1e-6:
            # At t=1, x_t = x_1, so any x_0 on the geodesic works
            # Return x_t as a reasonable choice
            return x_t

        # Ensure inputs are on the manifold
        # x_t = self.prior_dist.project_to_manifold(x_t)
        # x_1 = self.prior_dist.project_to_manifold(x_1)

        # x_0 = x_t @ expm((t/(t-1)) * logm(x_t^{-1} @ x_1))
        x_t_inv_x_1 = torch.linalg.lstsq(x_t, x_1, driver="gels").solution
        log_term = blogm(x_t_inv_x_1)
        x_0 = x_t @ expm((t / (t - 1.0)) * log_term)

        # Ensure x_0 is on manifold
        # x_0 = self.prior_dist.project_to_manifold(x_0)

        return x_0

    def _forward_simulate(self, x_0: Tensor, t_values: Tensor) -> tuple[Tensor, Tensor]:
        """Forward simulate from x_0 to get trajectories and log probabilities."""
        batch_size = x_0.shape[0]

        # Ensure x_0 is on manifold
        # x_0 = self.prior_dist.project_to_manifold(x_0)

        x_t = x_0
        x_t_all = [x_t]

        log_p_t = self.prior_dist.log_prob(x_0)
        log_p_all = [log_p_t]

        for i in range(len(t_values) - 1):
            t = t_values[i]
            delta_t = t_values[i + 1] - t_values[i]
            t_tensor = torch.full((batch_size, 1), t.item(), device=self.device)

            # Compute divergence
            with torch.set_grad_enabled(True):
                x_t_grad = x_t.detach().requires_grad_(True)
                div = batched_object_div(self.net.forward, x_t_grad, t_tensor)

            # Exponential Euler step
            A = self.net(x_t, t_tensor)
            x_t = x_t @ expm(delta_t * A)

            # Project back to manifold
            # x_t = self.prior_dist.project_to_manifold(x_t)

            log_p_t = log_p_t - delta_t * div

            x_t_all.append(x_t)
            log_p_all.append(log_p_t)

        x_t_all = torch.stack(x_t_all, dim=0)  # [n_steps + 1, batch, 2, 2]
        log_p_all = torch.stack(log_p_all, dim=0)

        return x_t_all, log_p_all

    def _integrate_divergence_to_time(self, x_0: Tensor, t: float, n_substeps: int) -> Tensor:
        """Integrate divergence from x_0 to time t."""
        batch_size = x_0.shape[0]
        delta_t = t / n_substeps

        x_t = x_0
        log_p_t = self.prior_dist.log_prob(x_0)  # [batch, M]

        for i in range(n_substeps):
            s = i * delta_t
            s_tensor = torch.full((batch_size, 1), s, device=self.device)

            # Compute divergence
            with torch.set_grad_enabled(True):
                x_t_grad = x_t.detach().requires_grad_(True)
                div = batched_object_div(self.net.forward, x_t_grad, s_tensor)

            # Exponential Euler step
            A = self.net(x_t, s_tensor)
            x_t = x_t @ expm(delta_t * A)
            log_p_t = log_p_t - delta_t * div

        # Return log probability at time t
        return log_p_t

    @torch.no_grad()
    def approx_p_x1_given_xt(
        self,
        x_0: torch.Tensor,
        p1_dist,  # a torch.distributions instance
        n_steps: int = 50,
        n_x1_grid: int = 100,
    ) -> dict[str, torch.Tensor]:
        """
        Compute p(x1 | x_t) for flow matching on prior group.
        """
        self.eval()
        device = x_0.device
        batch_size = x_0.shape[0]

        # Time grid
        if self.time_sampling == "uniform":
            # Evenly spaced timesteps
            t_values = torch.linspace(0.0, 1.0, n_steps + 1, device=self.device)
        elif self.time_sampling == "power":
            t_values = icdf_power(
                0,
                1,
                n_steps + 1,
                skewness=self.time_skewness,
                device=self.device,
            )
        elif self.time_sampling == "beta":
            t_values = icdf_beta(
                self.time_alpha,
                self.time_beta,
                n_steps + 1,
                device=self.device,
            )

        # Detect discrete target
        use_discrete = isinstance(p1_dist, DiscreteDeltaMixture) or isinstance(p1_dist, RandomDiscreteDeltaMixture)

        if use_discrete:
            # Discrete atoms: use matrix means directly
            base_obj = p1_dist.base_object.to(device)
            base_tfs = p1_dist.locs.to(device)

            x1_values = torch.einsum("mdd,nd->mnd", base_tfs, base_obj)  # [M, N, D]

            prior_weights = p1_dist.weights.to(device)
            log_prior_weights = torch.log(prior_weights)
            M = x1_values.shape[0]
        else:
            raise NotImplementedError()
            # # For continuous prior, sample from this distribution
            # x1_values = p1_dist.sample((n_x1_grid,)).to(device)  # [M, D, D]
            # M = x1_values.shape[0]
            #
            # log_p_x1_grid = p1_dist.log_prob(x1_values).view(M)

        event_shape = self.prior_dist.event_shape
        x_shape = x_0.shape[1:]

        # Storage tensors
        x_t_all = torch.zeros(n_steps + 1, batch_size, *x_shape, device=device)
        log_p_t_all = torch.zeros(n_steps + 1, batch_size, device=device)
        p_x1_given_xt_all = torch.zeros(n_steps + 1, batch_size, M, device=device)
        # For GMM, we also track component weights
        if not use_discrete and hasattr(p1_dist, "mixture_distribution"):
            K = p1_dist.mixture_distribution.probs.shape[0]
            weights_all = torch.zeros(n_steps + 1, batch_size, K, device=device)
        else:
            weights_all = torch.zeros(n_steps + 1, batch_size, M, device=device)

        # Forward simulate once for trajectories and marginal log p_t
        x_sim, logp_sim = self._forward_simulate(x_0, t_values)
        x_t_all[:] = x_sim
        log_p_t_all[:] = logp_sim

        # Main loop over times
        for i, t in enumerate(t_values):
            x_t = x_t_all[i]  # [batch, D, D]
            t_val = t.item()

            if t_val == 0.0:
                # At t=0, posterior equals prior
                if use_discrete:
                    post = prior_weights.unsqueeze(0).expand(batch_size, -1)  # [batch, K]
                    p_x1_given_xt_all[i] = post
                    weights_all[i] = post
                else:
                    p_x1 = torch.exp(log_p_x1_grid)
                    p_x1 = p_x1 / p_x1.sum()
                    p_x1_given_xt_all[i] = p_x1.unsqueeze(0).expand(batch_size, -1)

                    # Compute GMM component weights if applicable
                    if hasattr(p1_dist, "mixture_distribution"):
                        weights_all[i] = self._compute_gmm_weights_matrix(x1_values, p_x1.unsqueeze(0), p1_dist).expand(
                            batch_size, -1
                        )
                continue

            # For t > 0, compute likelihoods
            log_lik = torch.full((batch_size, M), -float("inf"), device=device)

            if t_val < 0.99:
                # Matrix geodesic inversion
                batch_M = batch_size * M
                x_t_exp = x_t.unsqueeze(1).expand(-1, M, -1, -1)  # [batch, M, N, D]
                x1_exp = x1_values.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch, M, N, D]

                x_t_flat = x_t_exp.reshape(batch_M, *x_shape)

                x1_flat = x1_exp.reshape(batch_M, *x_shape)

                # Compute x_0 for all pairs
                x0_all = self._matrix_geodesic_inverse(x_t_flat, x1_flat, t_val)

                # Compute log probability by integrating divergence
                log_p = self._integrate_divergence_to_time(x0_all, t_val, n_substeps=i)
                log_lik = log_p.view(batch_size, M)
            else:
                # Near t=1: use geodesic distance on prior group
                x_t_exp = x_t.unsqueeze(1).expand(-1, M, -1, -1)  # [batch, M, N, D]
                x1_exp = x1_values.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch, M, N, D]

                # Geodesic distance: ||log(x_t^{-1} @ x_1)||_F
                x_t_inv_x1 = torch.linalg.lstsq(
                    x_t_exp.reshape(batch_M, *x_shape),
                    x1_exp.reshape(batch_M, *x_shape),
                ).solution

                log_matrices = blogm(x_t_inv_x1).view(batch_size, M, event_shape[-1], event_shape[-1])
                distances = torch.norm(log_matrices.view(batch_size, M, -1), dim=-1)  # [batch, M]

                sigma = 0.1
                log_lik = -0.5 * (distances / sigma) ** 2

            # Combine with prior
            if use_discrete:
                log_post_unnorm = log_lik + log_prior_weights.unsqueeze(0)
            else:
                log_post_unnorm = log_lik + log_p_x1_grid.unsqueeze(0)

            # Normalize
            all_inf_mask = ~torch.isfinite(log_post_unnorm).any(dim=1)
            log_post_unnorm[all_inf_mask] = 0.0
            post = F.softmax(log_post_unnorm, dim=1)
            p_x1_given_xt_all[i] = post

            # Update weights
            if use_discrete:
                weights_all[i] = post
            elif hasattr(p1_dist, "mixture_distribution"):
                weights_all[i] = self._compute_gmm_weights_matrix(x1_values, post, p1_dist)

        # Extract component means as matrices
        if use_discrete:
            component_means = x1_values.clone()
        elif hasattr(p1_dist, "component_distribution"):
            # Convert GMM component means to matrices
            component_means = p1_dist.component_distribution.loc
        else:
            component_means = None

        return {
            "x_t_all": x_t_all,
            "log_p_t_all": log_p_t_all,
            "p_x1_given_xt_all": p_x1_given_xt_all,
            "weights_all": weights_all,
            "x1_grid": x1_values,
            "t_values": t_values,
            "component_means": component_means,  # [K, 2, 2]
            "use_discrete": use_discrete,
        }

    def _compute_gmm_weights_matrix(self, x1_matrices: Tensor, p_x1: Tensor, p1_dist) -> Tensor:
        """
        Compute posterior mixture weights for GMM components in GL+(2) space.

        Args:
            x1_matrices: [M, 2, 2] sampled matrices
            p_x1: [batch, M] posterior probabilities
            p1_dist: GMM distribution in GL+(2) space

        Returns:
            weights: [batch, K] component weights
        """
        if not hasattr(p1_dist, "mixture_distribution"):
            # Not a GMM, return p_x1 as weights
            return p_x1

        mix_probs = p1_dist.mixture_distribution.probs  # [K]
        K = mix_probs.shape[0]
        M = x1_matrices.shape[0]

        # Evaluate each component at each grid point
        log_mix = torch.log(mix_probs)  # [K]

        # For each component k, evaluate log p_k(x1_matrices)
        log_q = torch.zeros(K, M, device=x1_matrices.device)  # [K, M]

        for k in range(K):
            # Get k-th component distribution
            comp_k = p1_dist.component_distribution
            # Evaluate at all grid points
            log_q[k] = comp_k.log_prob(x1_matrices)  # [M]

        # Evaluate total log p1(x1_matrices)
        log_p1 = p1_dist.log_prob(x1_matrices)  # [M]

        # Compute ratio: p(k) * p(x|k) / p(x)
        log_ratio = log_mix.unsqueeze(1) + log_q - log_p1.unsqueeze(0)  # [K, M]
        ratio = torch.exp(log_ratio)

        # Compute weights: sum over grid points weighted by posterior
        weights_unnorm = p_x1 @ ratio.t()  # [batch, K]
        weights = weights_unnorm / (weights_unnorm.sum(dim=1, keepdim=True) + 1e-12)

        return weights
