"""
LSBO: Latent Space Bayesian Optimization.

Reference:
    Gómez-Bombarelli, R., et al. (2018). "Automatic Chemical Design Using a
    Data-Driven Continuous Representation of Molecules". ACS Central Science.

LSBO learns a latent space representation using a Variational Autoencoder (VAE)
and performs Bayesian optimization in this learned lower-dimensional latent space.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from torch import Tensor

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseOptimizer


class SimpleVAE(nn.Module):
    """
    Simple Variational Autoencoder for learning latent representations.

    Args:
        input_dim: Dimensionality of input
        latent_dim: Dimensionality of latent space
        hidden_dim: Hidden layer dimension
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


class LSBO(BaseOptimizer):
    """
    Latent Space Bayesian Optimization.

    Learns a VAE to map high-dimensional space to latent space,
    then performs BO in the latent space.

    Key Features:
    - Non-linear dimensionality reduction via VAE
    - Optimization in learned latent space
    - Can capture complex structure
    - Suitable for structured data (molecules, sequences, etc.)

    Args:
        input_dim: Dimensionality of input space
        bounds: Tensor of shape (2, input_dim)
        latent_dim: Latent space dimensionality (default: sqrt(d))
        vae_hidden_dim: VAE hidden layer size (default: 128)
        vae_train_steps: VAE training steps per update (default: 100)
        vae_lr: VAE learning rate (default: 0.001)
        pretrained_vae: Optional pre-trained VAE model
        device: Device for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        latent_dim: Optional[int] = None,
        vae_hidden_dim: int = 128,
        vae_train_steps: int = 100,
        vae_lr: float = 0.001,
        pretrained_vae: Optional[SimpleVAE] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(input_dim, bounds, device, dtype)

        # Default latent dim: sqrt(d)
        if latent_dim is None:
            latent_dim = max(2, int(input_dim ** 0.5))
        self.latent_dim = min(latent_dim, input_dim)

        self.vae_hidden_dim = vae_hidden_dim
        self.vae_train_steps = vae_train_steps
        self.vae_lr = vae_lr

        # Initialize VAE
        if pretrained_vae is not None:
            self.vae = pretrained_vae
        else:
            self.vae = SimpleVAE(input_dim, latent_dim, vae_hidden_dim)

        self.vae = self.vae.to(device=device, dtype=torch.float32)  # VAE uses float32

        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)

        self.model: Optional[SingleTaskGP] = None
        self.latent_bounds: Optional[Tensor] = None

    def _vae_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        mu: Tensor,
        logvar: Tensor
    ) -> Tensor:
        """
        Compute VAE loss (reconstruction + KL divergence).

        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance

        Returns:
            Total loss
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss

    def _train_vae(self):
        """Train VAE on observed data."""
        if self.X is None or len(self.X) < 10:
            # Not enough data to train VAE
            return

        X_train = self.X.float()  # VAE uses float32

        self.vae.train()
        for step in range(self.vae_train_steps):
            self.vae_optimizer.zero_grad()

            # Forward pass
            x_recon, mu, logvar = self.vae(X_train)

            # Compute loss
            loss = self._vae_loss(X_train, x_recon, mu, logvar)

            # Backward pass
            loss.backward()
            self.vae_optimizer.step()

        self.vae.eval()

    def _encode_to_latent(self, X: Tensor) -> Tensor:
        """
        Encode points to latent space.

        Args:
            X: Tensor of shape (n, input_dim)

        Returns:
            Tensor of shape (n, latent_dim)
        """
        self.vae.eval()
        with torch.no_grad():
            X_float = X.float()
            mu, _ = self.vae.encode(X_float)
            return mu.to(dtype=self.dtype)

    def _decode_from_latent(self, Z: Tensor) -> Tensor:
        """
        Decode latent points to input space.

        Args:
            Z: Tensor of shape (n, latent_dim)

        Returns:
            Tensor of shape (n, input_dim)
        """
        self.vae.eval()
        with torch.no_grad():
            Z_float = Z.float()
            X_recon = self.vae.decode(Z_float)
            X_recon = X_recon.to(dtype=self.dtype)

            # Clip to bounds
            X_recon = torch.clamp(X_recon, self.bounds[0], self.bounds[1])

            return X_recon

    def _estimate_latent_bounds(self) -> Tensor:
        """
        Estimate bounds for latent space based on encoded observations.

        Returns:
            Tensor of shape (2, latent_dim)
        """
        if self.X is None or len(self.X) == 0:
            # Default bounds
            return torch.stack([
                torch.full((self.latent_dim,), -3.0, device=self.device, dtype=self.dtype),
                torch.full((self.latent_dim,), 3.0, device=self.device, dtype=self.dtype)
            ])

        # Encode all observations
        Z = self._encode_to_latent(self.X)

        # Use empirical bounds with some margin
        margin = 1.0
        z_min = Z.min(dim=0).values - margin
        z_max = Z.max(dim=0).values + margin

        return torch.stack([z_min, z_max])

    def _fit_model(self):
        """Fit GP model in latent space."""
        # Train VAE on observations
        self._train_vae()

        # Encode observations to latent space
        Z = self._encode_to_latent(self.X)

        # Update latent bounds before fitting model
        self.latent_bounds = self._estimate_latent_bounds()

        # Standardize outputs manually to avoid BoTorch input warnings
        y_mean = self.y.mean()
        y_std = self.y.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        y_stdized = (self.y - y_mean) / y_std

        # Fit GP in latent space with transforms
        self.model = SingleTaskGP(
            train_X=Z,
            train_Y=y_stdized,
            input_transform=Normalize(d=self.latent_dim, bounds=self.latent_bounds),
            outcome_transform=None,
        ).to(device=self.device, dtype=self.dtype)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def suggest(self, n_suggestions: int = 1) -> Tensor:
        """
        Suggest next point(s) to evaluate.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Tensor of shape (n_suggestions, input_dim)
        """
        if self.X is None or len(self.X) == 0:
            raise ValueError("No observations yet. Call observe() first.")

        # Fit model (trains VAE and GP)
        self._fit_model()

        # Optimize acquisition function in latent space
        acq_func = ExpectedImprovement(
            model=self.model,
            best_f=self.y.max().item()
        )

        candidates_latent, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.latent_bounds,
            q=n_suggestions,
            num_restarts=10,
            raw_samples=512,
        )

        # Decode to input space
        candidates = self._decode_from_latent(candidates_latent)

        return candidates

    def observe(self, X: Tensor, y: Tensor):
        """
        Update optimizer with new observations.

        Args:
            X: Tensor of shape (n, input_dim)
            y: Tensor of shape (n, 1) or (n,)
        """
        X = X.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        if y.dim() == 1:
            y = y.unsqueeze(-1)

        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = torch.cat([self.X, X], dim=0)
            self.y = torch.cat([self.y, y], dim=0)

    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get best observed point."""
        if self.y is None or len(self.y) == 0:
            raise ValueError("No observations yet.")

        best_idx = self.y.argmax()
        return self.X[best_idx], self.y[best_idx]

    def get_vae(self) -> SimpleVAE:
        """Get trained VAE model."""
        return self.vae
