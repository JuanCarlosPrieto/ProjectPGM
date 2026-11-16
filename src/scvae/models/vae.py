import torch
from torch import Tensor, nn


class GaussianEncoder(nn.Module):
    """Approximate q(z | x) with a diagonal Gaussian."""

    def __init__(self, n_genes: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()

        self.hidden = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.log_std_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, counts: Tensor) -> tuple[Tensor, Tensor]:
        """Return the latent mean and log standard deviation for each cell."""
        hidden = self.hidden(counts)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden)
        return mean, log_std



def reparameterize(mean: Tensor, log_std: Tensor) -> Tensor:
    """Draw one sample per cell from its diagonal Gaussian posterior."""
    std = log_std.exp()
    epsilon = torch.randn_like(mean)
    return mean + std * epsilon



class PoissonDecoder(nn.Module):
    """Map each latent sample to one Poisson log-rate per gene."""

    def __init__(self, latent_dim: int, hidden_dim: int, n_genes: int) -> None:
        super().__init__()

        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.log_rate_head = nn.Linear(hidden_dim, n_genes)

    def forward(self, z: Tensor) -> Tensor:
        hidden = self.hidden(z)
        return self.log_rate_head(hidden)