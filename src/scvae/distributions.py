"""Element-wise log-likelihoods for single-cell RNA sequencing count data.

Each function returns a tensor of log-probabilities without reducing over genes
or averaging over cells. The VAE training objective handles these reductions.

Expected tensor shapes:
    counts:          [batch_size, n_genes]
    log_rate:        [batch_size, n_genes]          (Poisson)
    log_mean:        [batch_size, n_genes]          (Negative Binomial)
    log_dispersion:  [n_genes] or [batch_size, n_genes]

Log-parameterization ensures positive distribution parameters for finite inputs
and allows the decoder to output unconstrained real-valued parameters.
"""

import torch
from torch import Tensor
from torch.nn import functional as F


def poisson_log_prob(counts: Tensor, log_rate: Tensor) -> Tensor:
    """Compute log P(X=counts) for X ~ Poisson(rate=exp(log_rate)).

    Formula:
        k * log(rate) - rate - lgamma(k + 1)

    Input tensors must be broadcast-compatible. Counts must be non-negative
    integers, and log_rate must contain finite values.
    """
    counts = counts.to(dtype=log_rate.dtype, device=log_rate.device)

    return counts * log_rate - torch.exp(log_rate) - torch.lgamma(counts + 1.0)


def negative_binomial_log_prob(
    counts: Tensor,
    log_mean: Tensor,
    log_dispersion: Tensor,
) -> Tensor:
    """Compute log P(X=counts) for a Negative Binomial distribution.

    Parameterization:
        mean = exp(log_mean)
        dispersion = exp(log_dispersion)
        E[X] = mean
        Var[X] = mean + mean**2 / dispersion

    Log-probability:
        lgamma(k + dispersion) - lgamma(dispersion) - lgamma(k + 1)
        + dispersion * log(dispersion / (dispersion + mean))
        + k * log(mean / (dispersion + mean))

    ``torch.logaddexp`` computes log(dispersion + mean) more stably than
    summing two exponentials and then taking the logarithm.

    Input tensors must be broadcast-compatible. Counts must be non-negative
    integers, and both log-parameters must contain finite values.
    """
    counts = counts.to(dtype=log_mean.dtype, device=log_mean.device)
    log_dispersion = log_dispersion.to(
        dtype=log_mean.dtype,
        device=log_mean.device,
    )

    dispersion = torch.exp(log_dispersion)
    log_total = torch.logaddexp(log_mean, log_dispersion)

    log_coefficient = (
        torch.lgamma(counts + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(counts + 1.0)
    )
    log_probability = (
        dispersion * (log_dispersion - log_total)
        + counts * (log_mean - log_total)
    )

    return log_coefficient + log_probability
