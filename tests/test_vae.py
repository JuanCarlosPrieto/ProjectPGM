import torch

from scvae.distributions import poisson_log_prob
from scvae.models.vae import GaussianEncoder, PoissonDecoder, reparameterize
from scvae.losses import gaussian_kl_standard_normal, negative_elbo


def test_encoder_and_sampling() -> None:
    counts = torch.tensor(
        [[0, 4, 1, 7, 0],
         [3, 0, 2, 1, 5]],
        dtype=torch.float32,
    )
    encoder = GaussianEncoder(n_genes=5, hidden_dim=8, latent_dim=3)

    mean, log_std = encoder(counts)
    z = reparameterize(mean, log_std)

    # Two cells, three latent coordinates per cell.
    assert mean.shape == log_std.shape == z.shape == (2, 3)
    assert torch.isfinite(z).all()

    # Check that gradients pass through the sample to both encoder heads.
    z.sum().backward()
    assert encoder.mean_head.weight.grad is not None
    assert encoder.log_std_head.weight.grad is not None



def test_poisson_reconstruction() -> None:
    counts = torch.tensor(
        [[0, 4, 1, 7, 0],
         [3, 0, 2, 1, 5]],
        dtype=torch.float32,
    )
    encoder = GaussianEncoder(n_genes=5, hidden_dim=8, latent_dim=3)
    decoder = PoissonDecoder(latent_dim=3, hidden_dim=8, n_genes=5)

    mean, log_std = encoder(counts)
    z = reparameterize(mean, log_std)
    log_rate = decoder(z)

    log_prob_per_gene = poisson_log_prob(counts, log_rate)
    log_prob_per_cell = log_prob_per_gene.sum(dim=-1)

    assert log_rate.shape == counts.shape      # (2 cells, 5 genes)
    assert log_prob_per_cell.shape == (2,)     # One score per cell
    assert torch.isfinite(log_prob_per_cell).all()

    loss = negative_elbo(log_prob_per_gene, mean, log_std)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert encoder.mean_head.weight.grad is not None
    assert encoder.log_std_head.weight.grad is not None
    assert decoder.log_rate_head.weight.grad is not None



def test_gaussian_kl_known_values() -> None:
    mean = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    log_std = torch.zeros_like(mean)

    kl = gaussian_kl_standard_normal(mean, log_std)

    torch.testing.assert_close(kl, torch.tensor([0.0, 2.0]))