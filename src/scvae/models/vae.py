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



class PoissonVAE(nn.Module):
    """Variational autoencoder with a Poisson observation model."""

    def __init__(self, n_genes: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = GaussianEncoder(n_genes, hidden_dim, latent_dim)
        self.decoder = PoissonDecoder(latent_dim, hidden_dim, n_genes)

    def forward(self, counts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self.encoder(counts)
        z = reparameterize(mean, log_std)
        log_rate = self.decoder(z)
        return log_rate, mean, log_std



class NegativeBinomialDecoder(nn.Module):
    """Predict NB log-mean and log-dispersion for every gene."""

    def __init__(
        self, latent_dim: int, hidden_dim: int, n_genes: int
    ) -> None:
        super().__init__()

        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.log_mean_head = nn.Linear(hidden_dim, n_genes)
        self.log_dispersion_head = nn.Linear(hidden_dim, n_genes)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.hidden(z)
        log_mean = self.log_mean_head(hidden)
        log_dispersion = self.log_dispersion_head(hidden)
        return log_mean, log_dispersion



class NegativeBinomialVAE(nn.Module):
    """Variational autoencoder with an NB observation model."""

    def __init__(
        self, n_genes: int, hidden_dim: int, latent_dim: int
    ) -> None:
        super().__init__()

        self.encoder = GaussianEncoder(n_genes, hidden_dim, latent_dim)
        self.decoder = NegativeBinomialDecoder(
            latent_dim, hidden_dim, n_genes
        )

    def forward(
        self, counts: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mean, log_std = self.encoder(counts)
        z = reparameterize(mean, log_std)
        log_mean, log_dispersion = self.decoder(z)

        return log_mean, log_dispersion, mean, log_std