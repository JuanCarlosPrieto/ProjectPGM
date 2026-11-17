import torch
from torch import Tensor

from pathlib import Path

from scipy.io import mmread
from scipy.sparse import csr_matrix


def make_toy_counts(seed: int = 42) -> Tensor:
    """Generate counts for 256 cells and 20 genes."""
    generator = torch.Generator().manual_seed(seed)

    # Each row contains the expected counts for one cell group.
    rates = torch.ones(2, 20)
    rates[0, :10] = 4.0
    rates[1, 10:] = 4.0

    # Alternate between the two groups, then sample counts for each cell.
    groups = torch.arange(256) % 2
    return torch.poisson(rates[groups], generator=generator)



def load_10x_counts(directory: Path) -> csr_matrix:
    """Load a 10x Matrix Market file as a sparse cells-by-genes matrix."""
    genes_by_cells = mmread(directory / "matrix.mtx")
    cells_by_genes = genes_by_cells.T.tocsr()
    return cells_by_genes.astype("float32")