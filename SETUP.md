# Development Environment Setup

This project supports both **uv** (recommended) and **conda** workflows.

## Option 1: uv (Recommended - Fast & Modern)

### Initial Setup

```bash
# Create virtual environment (correct command!)
uv venv

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install project in editable mode with all dependencies
uv pip install -e ".[dev]"

# Install Jupyter kernel
uv pip install ipykernel
python -m ipykernel install --user --name=siglet_architecture
```

### Daily Usage

```bash
# Activate environment
source .venv/bin/activate

# Launch Jupyter
jupyter notebook notebooks/Siglet-Qubit-Simulation.ipynb

# Run any scripts
python -m siglet_architecture.train
```

### Sync Dependencies

```bash
# After updating pyproject.toml
uv pip sync pyproject.toml
```

---

## Option 2: Conda (Traditional Data Science)

### Initial Setup

```bash
# Create environment from file
conda env create -f environment.yml

# Activate
conda activate siglet_architecture
```

### Daily Usage

```bash
conda activate siglet_architecture
jupyter notebook notebooks/Siglet-Qubit-Simulation.ipynb
```

---

## Why uv?

1. **10-100x faster** than pip/conda for installs
2. **Better dependency resolution** with lockfiles
3. **Deterministic builds** across machines
4. **Native ARM64 support** on M1/M2 Macs
5. All scientific packages (numpy, scipy, sklearn) have excellent wheels now

## Why Conda?

1. **More conservative** - battle-tested in data science
2. **Better for complex C/Fortran dependencies** (though less relevant now)
3. **Broader ecosystem** with conda-forge

---

## Recommended: uv for M2 MacBook Air

Your M2 Mac will benefit significantly from uv's speed and native ARM64 wheels.
