# Siglet-Qubit Architecture


## Project Overview

This project implements a classical prototype of the Siglet-Qubit kernel, a system for quantum information processing that explores stability, resonance, and coherence in quantum-like systems. The architecture investigates how various parameter combinations affect truth scores and decay curves, identifying stable regions in parameter space for potential quantum computing applications.

## Objectives

- Build a classical simulation of the Siglet-Qubit system
- Generate and analyze decay curves with varying parameters (θ, τ)
- Identify regions of parameter space with similar behavior using clustering techniques
- Measure curve similarities using Dynamic Time Warping (DTW)
- Evaluate the effects of noise on cluster stability
- Extract scientific findings about shape invariance in the system
- Establish an optimal set of primitive operators for the system

## Technical Approach

The project employs several data science techniques:

- **Parameter Space Exploration**: Systematic generation of decay curves across theta-tau grids
- **Time Series Analysis**: DTW distance metrics for measuring curve similarity
- **Clustering**: DBSCAN and Spectral Clustering to identify regions with similar dynamics
- **Optimization**: Optuna for hyperparameter tuning and operator selection
- **Noise Analysis**: Stability testing with various noise levels
- **Visualization**: Heatmaps, cluster plots, and 3D visualizations to interpret results
- **Experiment Tracking**: MLflow to record parameters, metrics, and artifacts

## Key Features

- `SigletQubit` class with configurable parameters (θ, τ, μ, etc.)
- Truth score calculation with temporal decay
- Parameter space visualization with heatmaps
- Comprehensive clustering analysis with DBSCAN and Spectral Clustering
- Shape invariance analysis of decay curves
- Noise resilience evaluation
- Primitive operator selection and optimization
- Full MLflow integration for reproducible experiments

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks, including Siglet-Qubit-Simulation.ipynb
│
├── pyproject.toml     <- Project configuration file with package metadata and tool configuration
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── setup.cfg          <- Configuration file for flake8
│
└── siglet_architecture   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes siglet_architecture a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── models                  <- Module containing model implementations
    │   ├── __init__.py
    │   ├── siglet_qubit.py     <- SigletQubit class implementation
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Luiz-Frias/siglet_architecture.git
cd siglet_architecture

# Set up the environment
conda env create -f environment.yml
conda activate siglet_architecture

# Run the notebook
jupyter notebook notebooks/Siglet-Qubit-Simulation.ipynb
```

## Technology Stack

- Python 3.10+
- Numpy, Scipy, Pandas
- Scikit-learn for machine learning algorithms
- MLflow for experiment tracking
- Matplotlib, Seaborn for visualization
- FastDTW, tslearn for time series analysis
- Optuna for hyperparameter optimization

--------
