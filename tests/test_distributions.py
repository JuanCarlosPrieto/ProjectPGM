"""Tests for count log-likelihoods and their differentiability."""

import math

import torch

from scvae.distributions import negative_binomial_log_prob, poisson_log_prob


def test_poisson_matches_pytorch_reference() -> None:
    counts = torch.tensor([[0, 2, 5], [1, 0, 3]], dtype=torch.float64)
    rates = torch.tensor([[0.5, 2.0, 4.5], [1.0, 0.8, 3.0]], dtype=torch.float64)
    log_rate = rates.log()

    actual = poisson_log_prob(counts, log_rate)
    expected = torch.distributions.Poisson(rate=rates).log_prob(counts)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == counts.shape


def test_poisson_zero_count() -> None:
    log_rate = torch.tensor([math.log(2.0)], dtype=torch.float64)
    actual = poisson_log_prob(torch.tensor([0]), log_rate)

    torch.testing.assert_close(actual, torch.tensor([-2.0], dtype=torch.float64))


def test_negative_binomial_matches_pytorch_reference() -> None:
    counts = torch.tensor([[0, 1, 9], [3, 0, 2]], dtype=torch.float64)
    means = torch.tensor([[0.5, 1.5, 4.0], [3.0, 2.0, 1.0]], dtype=torch.float64)
    dispersions = torch.tensor([2.0, 5.0, 0.7], dtype=torch.float64)
    log_mean = means.log()
    log_dispersion = dispersions.log()

    actual = negative_binomial_log_prob(counts, log_mean, log_dispersion)

    # PyTorch's NegativeBinomial uses logits = log(mean / dispersion).
    expected = torch.distributions.NegativeBinomial(
        total_count=dispersions,
        logits=log_mean - log_dispersion,
    ).log_prob(counts)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == counts.shape


def test_negative_binomial_approaches_poisson_when_dispersion_grows() -> None:
    counts = torch.arange(0, 10, dtype=torch.float64)
    log_mean = torch.tensor(math.log(3.0), dtype=torch.float64)
    log_dispersion = torch.tensor(math.log(1e6), dtype=torch.float64)

    actual = negative_binomial_log_prob(counts, log_mean, log_dispersion)
    expected = poisson_log_prob(counts, log_mean)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_negative_binomial_gradients_are_finite() -> None:
    counts = torch.tensor([[0.0, 3.0, 8.0]])
    log_mean = torch.tensor([[0.0, 1.0, 1.8]], requires_grad=True)
    log_dispersion = torch.tensor([0.0, -0.5, 0.5], requires_grad=True)

    loss = -negative_binomial_log_prob(counts, log_mean, log_dispersion).sum()
    loss.backward()

    assert log_mean.grad is not None
    assert log_dispersion.grad is not None
    assert torch.isfinite(log_mean.grad).all()
    assert torch.isfinite(log_dispersion.grad).all()


def test_poisson_gradients_are_finite() -> None:
    counts = torch.tensor([[0.0, 3.0, 8.0]])
    log_rate = torch.tensor([[0.0, 1.0, 1.8]], requires_grad=True)

    loss = -poisson_log_prob(counts, log_rate).sum()
    loss.backward()

    assert log_rate.grad is not None
    assert torch.isfinite(log_rate.grad).all()


def test_reconstruction_reduction_is_explicit() -> None:
    counts = torch.tensor([[0.0, 2.0], [1.0, 3.0]])
    log_rate = torch.zeros_like(counts)

    elementwise_log_prob = poisson_log_prob(counts, log_rate)
    assert elementwise_log_prob.shape == (2, 2)

    # Sum over genes and then average across the batch (one scalar loss).
    reconstruction_nll = -elementwise_log_prob.sum(dim=-1).mean()
    assert reconstruction_nll.ndim == 0
    assert torch.isfinite(reconstruction_nll)
