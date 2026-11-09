# scVAE — reimplementación pedagógica

Re-implementation of the article by Grønbech et al. (2020),
*scVAE: variational auto-encoders for single-cell gene expression data*.

## Requirements

- Python 3.12

## Environment preparation (Linux / macOS / WSL)

```bash
python3.12 -m venv .venv
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


