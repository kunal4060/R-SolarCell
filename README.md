# R-SolarCell: Machine Learning for Solar Cell Performance Prediction

> **Predict solar cell efficiency using ML & Neural Networks from 3,000 simulated samples**

---

## 🚀 Quick Start

**Want to run the models right now?** → See [02_QUICK_START_GUIDE.md](05_documentation/02_QUICK_START_GUIDE.md)

```bash
# Install dependencies
pip install numpy pandas scikit-learn xgboost matplotlib seaborn

# Run ML models
cd 02_ml_models
python train_ml_models.py

# Run Neural Networks
cd 03_neural_networks
python train_neural_networks.py
```

---

## 📚 Documentation

All documentation is organized in the **[05_documentation/](05_documentation/)** folder:

| # | Document | Description | Reading Time |
|---|----------|-------------|--------------|
| 00 | [Documentation Index](05_documentation/00_DOCUMENTATION_INDEX.md) | Navigation guide for all docs | 2 min |
| 01 | [Project Overview](05_documentation/01_README_PROJECT_OVERVIEW.md) | **Start here** - Complete project guide | 20 min |
| 02 | [Quick Start Guide](05_documentation/02_QUICK_START_GUIDE.md) | Run models in 5 minutes | 5 min |
| 03 | [Model Performance](05_documentation/03_MODEL_PERFORMANCE_SUMMARY.md) | R² scores and comparisons | 3 min |
| 04 | [Data Dictionary](05_documentation/04_DATA_DICTIONARY.md) | Dataset columns explained | 10 min |
| 05 | [Project Structure](05_documentation/05_PROJECT_STRUCTURE.md) | File organization guide | 5 min |

---

## 📂 Project Structure

```
R-SolarCell/
│
├── 01_dataset/                    ← Training data (3,000 samples)
│   └── solar_cell_dataset_3000_samples.csv
│
├── 02_ml_models/                  ← Traditional ML training scripts
│   └── train_ml_models.py         (XGBoost, RF, GB, Ridge, KNN)
│
├── 03_neural_networks/            ← Neural network training scripts
│   └── train_neural_networks.py   (ANN, CNN, RNN)
│
├── 04_results/                    ← All model outputs & plots
│   ├── ml_model_results/          (ML metrics & predictions)
│   ├── nn_model_results/          (NN metrics & predictions)
│   ├── all_model_outputs_combined.csv
│   └── all_model_performance_comparison.csv
│
├── 05_documentation/              ← All documentation (read here!)
│   ├── 00_DOCUMENTATION_INDEX.md
│   ├── 01_README_PROJECT_OVERVIEW.md
│   ├── 02_QUICK_START_GUIDE.md
│   ├── 03_MODEL_PERFORMANCE_SUMMARY.md
│   ├── 04_DATA_DICTIONARY.md
│   └── 05_PROJECT_STRUCTURE.md
│
└── 06_utilities/                  ← Helper scripts (future use)
```

---

## 📊 Quick Results

### Best Model Performance

| Model | Target | R² Score | Status |
|-------|--------|----------|--------|
| **XGBoost** | Jsc | **0.962** | ⭐ Excellent |
| **XGBoost** | η | **0.890** | ⭐ Excellent |
| **XGBoost** | Voc | **0.801** | ✅ Good |
| **ANN** | Overall | **0.82** | ✅ Good |

### Dataset Summary
- **Samples**: 3,000 simulated solar cells
- **Input Features**: 9 (layer thickness, bandgap, electron affinity × 3 layers)
- **Target Variables**: 4 (Voc, Jsc, FF, η)
- **Simulation**: SCAPS 1-D with Latin Hypercube Sampling

---

## 🔬 What This Project Does

1. **Trains ML models** to predict solar cell performance from material parameters
2. **Compares 8 different models** (5 ML + 3 Neural Networks)
3. **Achieves R² > 0.8** for Voc, Jsc, and η predictions
4. **Generates 40+ professional plots** for analysis
5. **Provides physical insights** into solar cell behavior

---

## 🛠️ Installation

### Requirements
- Python 3.8+
- numpy, pandas, scikit-learn, xgboost, matplotlib, seaborn

### Install Command
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn tensorflow keras
```

---

## 📖 Citation

If you use this work, please cite:

```

---


---

**Last Updated**: April 26, 2026  
**Version**: 1.0
