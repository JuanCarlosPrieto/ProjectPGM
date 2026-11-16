from torch import Tensor


def gaussian_kl_standard_normal(mean: Tensor, log_std: Tensor) -> Tensor:
    """Return the KL divergence to N(0, I) for each cell."""
    variance = (2 * log_std).exp()
    kl_per_dim = 0.5 * (mean.square() + variance - 1 - 2 * log_std)
    return kl_per_dim.sum(dim=-1)


def negative_elbo(
    log_prob_per_gene: Tensor,
    mean: Tensor,
    log_std: Tensor,
) -> Tensor:
    """Return the batch-averaged negative ELBO using one latent sample per cell."""
    reconstruction_nll = -log_prob_per_gene.sum(dim=-1)
    kl = gaussian_kl_standard_normal(mean, log_std)
    return (reconstruction_nll + kl).mean()