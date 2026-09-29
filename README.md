# R-SolarCell

Machine learning for solar cell performance prediction. The project trains classical ML models and neural networks on simulated solar cell data to predict efficiency (η) and related photovoltaic parameters from device inputs.

## What it does

The repository is organized as a full experiment pipeline:

1. Load the simulated solar cell dataset from `01_dataset/`.
2. Train classical ML models (XGBoost, Random Forest, Gradient Boosting, Ridge, KNN) with `02_ml_models/train_ml_models.py`.
3. Train neural networks (ANN, CNN, RNN) with `03_neural_networks/train_neural_networks.py`.
4. Compare metrics and plots saved under `04_results/`.
5. Read the full write-up in `05_documentation/`, starting with `01_README_PROJECT_OVERVIEW.md`.

## Requirements

- Python 3.8 or newer
- `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`
- A terminal, or [Visual Studio Code](https://code.visualstudio.com/)

## Install

### 1. Download the project

```bash
git clone https://github.com/kunal4060/R-SolarCell.git
cd R-SolarCell
```

### 2. Install the Python packages

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

## Usage

Train the classical ML models:

```bash
cd 02_ml_models
python train_ml_models.py
```

Train the neural networks:

```bash
cd 03_neural_networks
python train_neural_networks.py
```

Results (metrics CSVs and comparison plots) are written to `04_results/`. For a guided walkthrough, start with `05_documentation/02_QUICK_START_GUIDE.md`.

## Project structure

```text
R-SolarCell/
├── 01_dataset/              ← training data (solar_cell_dataset_3000_samples.csv)
├── 02_ml_models/           ← train_ml_models.py (XGBoost, RF, GB, Ridge, KNN)
├── 03_neural_networks/     ← train_neural_networks.py (ANN, CNN, RNN)
├── 04_results/             ← metrics, predictions, and plots
└── 05_documentation/       ← full docs; start with 01_README_PROJECT_OVERVIEW.md
```

## License

Provided for learning and research purposes. No warranty is provided.
