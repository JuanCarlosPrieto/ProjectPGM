# scVAE — reimplementación pedagógica

Re-implementation of the article by Grønbech et al. (2020),
*scVAE: variational auto-encoders for single-cell gene expression data*.

## Requirements

- Python 3.12

## Environment preparation (Linux / macOS / WSL)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Environment preparation (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code: `Python: Select Interpreter` -> `.venv`.

## Installation verification

```bash
python -c "import sys, torch, numpy, scipy, pytest; print(sys.version); print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m pip check
```

## Reproducibility
If versions are modified, please...

```bash
python -m pip freeze > requirements-lock.txt
```

## Dataset

Our initial experiments use the CD19+ B-cell dataset from 10x Genomics,
one of the nine populations included in the scVAE paper's PBMC dataset.

The count matrix contains 10,085 cells and 32,738 genes.

Source:
https://www.10xgenomics.com/datasets/cd-19-plus-b-cells-1-standard-1-1-0

Run the following commands from the repository root on Linux:

```bash
mkdir -p data/raw/10x/b_cells

curl -fL --retry 3 \
  https://cf.10xgenomics.com/samples/cell-exp/1.1.0/b_cells/b_cells_filtered_gene_bc_matrices.tar.gz \
  -o data/raw/10x/b_cells/b_cells_filtered_gene_bc_matrices.tar.gz

tar -xzf data/raw/10x/b_cells/b_cells_filtered_gene_bc_matrices.tar.gz \
  -C data/raw/10x/b_cells
```

The extracted files are located in
`data/raw/10x/b_cells/filtered_matrices_mex/hg19/`:

- `matrix.mtx` contains the sparse gene-by-cell count matrix.
- `genes.tsv` contains gene identifiers and names.
- `barcodes.tsv` contains cell identifiers.

The loader transposes the matrix to cells × genes.
Training uses raw counts as reconstruction targets and applies
`log1p` to the encoder input.

## Run the experiments

Activate the environment and run all commands from the repository root:

```bash
source .venv/bin/activate
python -m pytest -q

OMP_NUM_THREADS=2 PYTHONPATH=src python -m scvae.training
OMP_NUM_THREADS=2 PYTHONPATH=src python -m scvae.training_nb

PYTHONPATH=src python -m scvae.plot_training
```

Both models use the same cell split, with seed 7 and approximate
training/validation/test proportions of 81%/9%/10%.

The current configuration trains each model for 20 epochs.
The checkpoint with the lowest observed validation negative ELBO
is saved.

## Generated files

Training produces:

- `checkpoints/poisson_b_cells_best.pt`
- `checkpoints/poisson_b_cells_history.pt`
- `checkpoints/nb_b_cells_best.pt`
- `checkpoints/nb_b_cells_history.pt`

Plotting produces:

- `results/poisson_vs_nb_training.png`

The `data/`, `checkpoints/`, and `results/` directories are excluded from Git. Reproduce their contents using the commands above. Numerical results can vary across software versions and hardware.
