import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split

from scvae.data import load_10x_counts
from scvae.distributions import poisson_log_prob
from scvae.losses import negative_elbo
from scvae.models.vae import PoissonVAE

from pathlib import Path
from scvae.losses import gaussian_kl_standard_normal


def batch_loss(model: PoissonVAE, counts: Tensor) -> Tensor:
    """Compute the mean negative ELBO on the original gene counts."""
    encoder_input = torch.log1p(counts)
    log_rate, mean, log_std = model(encoder_input)

    log_prob = poisson_log_prob(counts, log_rate)
    return negative_elbo(log_prob, mean, log_std)


@torch.no_grad()
@torch.no_grad()
def evaluate(model: PoissonVAE, loader: DataLoader) -> dict[str, float]:
    """Estimate reconstruction NLL and KL, averaged over cells."""
    model.eval()
    total_reconstruction = 0.0
    total_kl = 0.0

    for counts in loader:
        log_rate, mean, log_std = model(torch.log1p(counts))

        log_prob = poisson_log_prob(counts, log_rate)
        reconstruction = -log_prob.sum(dim=-1)
        kl = gaussian_kl_standard_normal(mean, log_std)

        total_reconstruction += reconstruction.sum().item()
        total_kl += kl.sum().item()

    n_cells = len(loader.dataset)
    reconstruction = total_reconstruction / n_cells
    kl = total_kl / n_cells

    return {
        "negative_elbo": reconstruction + kl,
        "reconstruction": reconstruction,
        "kl": kl,
    }



@torch.no_grad()
def evaluate_poisson_baseline(
    counts,
    train_indices: list[int],
    val_loader: DataLoader,
) -> float:
    """Evaluate a fixed Poisson rate per gene on held-out cells."""
    mean_per_gene = counts[train_indices].mean(axis=0)
    rates = torch.as_tensor(mean_per_gene, dtype=torch.float32).reshape(-1)

    # A small floor keeps log-rates finite for genes unseen in training.
    log_rates = rates.clamp_min(1e-8).log()

    total_nll = 0.0
    for batch in val_loader:
        log_prob = poisson_log_prob(batch, log_rates.expand_as(batch))
        nll_per_cell = -log_prob.sum(dim=1)
        total_nll += nll_per_cell.sum().item()

    return total_nll / len(val_loader.dataset)



def main() -> None:
    torch.manual_seed(42)
    
    directory = Path("data/raw/10x/b_cells/filtered_matrices_mex/hg19")
    counts = load_10x_counts(directory)
    n_cells, n_genes = counts.shape

    # Split cell indices so the count matrix stays sparse.
    n_train = int(0.81 * n_cells)
    n_val = int(0.09 * n_cells)
    n_test = n_cells - n_train - n_val

    train_set, val_set, test_set = random_split(
        range(n_cells),
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(7),
    )

    def collate_cells(cell_indices: list[int]) -> Tensor:
        """Convert only the selected cells to a dense float32 tensor."""
        return torch.from_numpy(counts[cell_indices].toarray())

    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True, collate_fn=collate_cells
    )
    val_loader = DataLoader(
        val_set, batch_size=32, shuffle=False, collate_fn=collate_cells
    )

    print(f"Cells: {n_cells} | Genes: {n_genes}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    model = PoissonVAE(n_genes=counts.shape[1], hidden_dim=32, latent_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    baseline_nll = evaluate_poisson_baseline(
        counts, train_set.indices, val_loader
    )
    print(f"Poisson baseline | val NLL: {baseline_nll:.2f}")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "poisson_b_cells_best.pt"

    initial_metrics = evaluate(model, val_loader)
    print(
        f"Before training | "
        f"val negative ELBO: {initial_metrics['negative_elbo']:.2f}"
    )

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            loss = batch_loss(model, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)

        train_loss = total_loss / len(train_set)
        metrics = evaluate(model, val_loader)
        val_loss = metrics["negative_elbo"]
        gain = baseline_nll - val_loss

        history.append({
            "epoch": epoch,
            "train_negative_elbo": train_loss,
            "val_negative_elbo": val_loss,
            "val_reconstruction": metrics["reconstruction"],
            "val_kl": metrics["kl"],
        })

        print(
            f"Epoch {epoch:02d} | train: {train_loss:.2f} "
            f"| val: {val_loss:.2f} "
            f"| reconstruction: {metrics['reconstruction']:.2f} "
            f"| KL: {metrics['kl']:.2f} "
            f"| gain: {gain:+.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_genes": n_genes,
                    "hidden_dim": 32,
                    "latent_dim": 2,
                },
                "encoder_transform": "log1p",
                "data_directory": str(directory),
                "train_indices": list(train_set.indices),
                "val_indices": list(val_set.indices),
                "test_indices": list(test_set.indices),
                "val_metrics": metrics,
                "baseline_nll": baseline_nll,
            }, checkpoint_path)

            print("  Saved best model")

    torch.save(history, checkpoint_dir / "poisson_b_cells_history.pt")
    print(f"Best validation negative ELBO: {best_val_loss:.2f}")


if __name__ == "__main__":
    main()