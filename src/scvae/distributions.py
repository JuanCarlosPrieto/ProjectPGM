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


def _zero_inflated_log_prob(
    counts: Tensor,
    base_log_prob: Tensor,
    zero_logits: Tensor,
) -> Tensor:
    """Add a Bernoulli structural-zero component to a count distribution."""
    zero_logits = zero_logits.to(
        dtype=base_log_prob.dtype,
        device=base_log_prob.device,
    )

    log_pi = F.logsigmoid(zero_logits)
    log_one_minus_pi = F.logsigmoid(-zero_logits)

    regular_log_prob = log_one_minus_pi + base_log_prob

    zero_log_prob = torch.logaddexp(
        log_pi,
        regular_log_prob,
    )

    return torch.where(
        counts == 0,
        zero_log_prob,
        regular_log_prob,
    )


def zero_inflated_poisson_log_prob(
    counts: Tensor,
    log_rate: Tensor,
    zero_logits: Tensor,
) -> Tensor:
    """Compute element-wise Zero-Inflated Poisson log-probabilities."""
    base_log_prob = poisson_log_prob(counts, log_rate)

    return _zero_inflated_log_prob(
        counts,
        base_log_prob,
        zero_logits,
    )


def zero_inflated_negative_binomial_log_prob(
    counts: Tensor,
    log_mean: Tensor,
    log_dispersion: Tensor,
    zero_logits: Tensor,
) -> Tensor:
    """Compute element-wise Zero-Inflated Negative Binomial log-probabilities."""
    base_log_prob = negative_binomial_log_prob(
        counts,
        log_mean,
        log_dispersion,
    )

    return _zero_inflated_log_prob(
        counts,
        base_log_prob,
        zero_logits,
    )



def constrained_poisson_log_prob(
    counts: Tensor,
    gene_logits: Tensor,
    sequencing_depth: Tensor,
) -> Tensor:
    """Compute element-wise Constrained Poisson log-probabilities.

    Args:
        counts: Observed counts [batch_size, n_genes].
        gene_logits: Unnormalized gene probabilities [batch_size, n_genes].
        sequencing_depth: Positive depth values [batch_size] or [batch_size, 1].

    Returns:
        Element-wise log-probabilities [batch_size, n_genes].
    """
    sequencing_depth = sequencing_depth.to(
        dtype=gene_logits.dtype,
        device=gene_logits.device,
    )

    if sequencing_depth.ndim == 1:
        sequencing_depth = sequencing_depth.unsqueeze(-1)

    log_gene_probs = F.log_softmax(
        gene_logits,
        dim=-1,
    )

    log_rate = sequencing_depth.log() + log_gene_probs

    return poisson_log_prob(counts, log_rate)



def piecewise_categorical_poisson_log_prob(
    counts: Tensor,
    category_logits: Tensor,
    log_tail_rate: Tensor,
    cutoff: int = 4,
) -> Tensor:
    """Compute element-wise Piece-wise Categorical Poisson log-probabilities.

    Args:
        counts: Observed counts [batch_size, n_genes].
        category_logits: Category logits [batch_size, n_genes, cutoff + 1].
        log_tail_rate: Log Poisson tail rates [batch_size, n_genes].
        cutoff: First count modeled by the shifted Poisson tail.

    Returns:
        Element-wise log-probabilities [batch_size, n_genes].
    """
    if cutoff < 1 or category_logits.shape[-1] != cutoff + 1:
        raise ValueError(
            "Expected cutoff >= 1 and cutoff + 1 category logits."
        )

    log_weights = F.log_softmax(
        category_logits,
        dim=-1,
    )

    indices = counts.long().clamp(max=cutoff).unsqueeze(-1)

    log_category_prob = log_weights.gather(
        -1,
        indices,
    ).squeeze(-1)

    tail_counts = (counts - cutoff).clamp_min(0)

    tail_log_prob = poisson_log_prob(
        tail_counts,
        log_tail_rate,
    )

    return log_category_prob + torch.where(
        counts >= cutoff,
        tail_log_prob,
        torch.zeros_like(tail_log_prob),
    )
