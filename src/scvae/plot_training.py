from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main() -> None:
    checkpoint_dir = Path("checkpoints")
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "Poisson": "poisson_b_cells",
        "NB": "nb_b_cells",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for label, prefix in runs.items():
        history = torch.load(
            checkpoint_dir / f"{prefix}_history.pt",
            map_location="cpu",
            weights_only=True,
        )
        checkpoint = torch.load(
            checkpoint_dir / f"{prefix}_best.pt",
            map_location="cpu",
            weights_only=True,
        )

        epochs = [row["epoch"] for row in history]
        train = [row["train_negative_elbo"] for row in history]
        val = [row["val_negative_elbo"] for row in history]
        reconstruction = [row["val_reconstruction"] for row in history]
        kl = [row["val_kl"] for row in history]

        line, = axes[0].plot(epochs, val, label=f"{label} validation")
        axes[0].plot(
            epochs,
            train,
            linestyle="--",
            color=line.get_color(),
            alpha=0.7,
            label=f"{label} training",
        )
        axes[1].plot(epochs, reconstruction, label=label)
        axes[2].plot(epochs, kl, label=label)

        metrics = checkpoint["val_metrics"]
        print(
            f"{label} | best epoch: {checkpoint['epoch']} "
            f"| validation negative ELBO: "
            f"{metrics['negative_elbo']:.2f} "
            f"| reconstruction: {metrics['reconstruction']:.2f} "
            f"| KL: {metrics['kl']:.2f}"
        )

    titles = [
        "Negative ELBO",
        "Validation reconstruction NLL",
        "Validation KL",
    ]

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Nats per cell")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    output_path = output_dir / "poisson_vs_nb_training.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to {output_path}")



if __name__ == "__main__":
    main()