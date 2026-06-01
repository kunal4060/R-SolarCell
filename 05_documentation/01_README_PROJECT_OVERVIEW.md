# R-SolarCell: Machine Learning Model for Solar Cell Performance Prediction

## 📋 Project Overview

This project uses advanced machine learning techniques to predict the performance of thin-film solar cells based on material parameters. The models predict four critical solar cell performance metrics: **Open-Circuit Voltage (Voc)**, **Short-Circuit Current (Jsc)**, **Fill Factor (FF)**, and **Power Conversion Efficiency (η)** from 9 input parameters describing the physical properties of a 3-layer solar cell structure.

### Solar Cell Architecture
```
Layer 1: PEDOT-HTL (Hole Transport Layer - p-type)
Layer 2: N-CS2AgBiBr6 (n-type perovskite)
Layer 3: P-CsPb(I0.6Br0.4)3 (p-type perovskite absorber)
```

---

## 📊 Dataset Description

### Data Source
- **Total Samples**: 3,000 simulated solar cell configurations
- **Simulation Method**: SCAPS 1-D (Solar Cell Capacitance Simulator)
- **Data Generation**: Latin Hypercube Sampling (LHS) for comprehensive parameter space coverage
- **Physical Validation**: All samples validated within realistic operating ranges

### Input Features (9 Parameters)

| Layer | Parameter | Unit | Min | Max | Description |
|-------|-----------|------|-----|-----|-------------|
| **L1: PEDOT-HTL** | Thickness | µm | 0.05 | 0.60 | Hole transport layer thickness |
| | Band Gap | eV | 1.20 | 1.80 | Energy bandgap |
| | Electron Affinity | eV | 3.40 | 3.80 | Electron affinity |
| **L2: N-CS2AgBiBr6** | Thickness | µm | 0.20 | 1.00 | n-type perovskite thickness |
| | Band Gap | eV | 1.80 | 2.40 | Energy bandgap |
| | Electron Affinity | eV | 3.60 | 4.00 | Electron affinity |
| **L3: P-CsPb(I0.6Br0.4)3** | Thickness | µm | 0.20 | 1.20 | p-type absorber thickness |
| | Band Gap | eV | 1.60 | 2.10 | Energy bandgap |
| | Electron Affinity | eV | 3.60 | 4.00 | Electron affinity |

### Target Variables (4 Performance Metrics)

| Metric | Symbol | Unit | Range | Description |
|--------|--------|------|-------|-------------|
| **Open-Circuit Voltage** | Voc | V | 0.96 - 1.46 | Maximum voltage without current flow |
| **Short-Circuit Current** | Jsc | mA/cm² | 12.1 - 29.8 | Maximum current at zero voltage |
| **Fill Factor** | FF | % | 74.3 - 81.8 | Measure of IV curve "squareness" |
| **Efficiency** | η | % | 11.8 - 22.0 | Overall power conversion efficiency |

---

## 📁 Project Structure

```
R-SolarCell/
│
├── README.md                              # This documentation file
│
├── ml_training/                            # Traditional ML models
│   ├── training.py                         # Main training script
│   ├── part1_2_3_4_6_7_8_9_10_combined.csv # Dataset (3000 samples)
│   └── results_part1_2_3_4_6_7_8_9_10/    # Results directory
│       ├── model_metrics.csv               # Performance metrics summary
│       ├── test_set_predictions.csv        # Model predictions on test set
│       ├── model_performance_r2.png        # R² comparison bar chart
│       ├── model_performance_rmse.png      # RMSE comparison bar chart
│       ├── model_performance_mae.png       # MAE comparison bar chart
│       ├── model_performance_overall.png   # Overall model performance
│       └── predicted_vs_actual_*.png       # Scatter plots (16 total)
│
├── nural_network/                          # Neural network training
│   ├── neural_network_training.py          # ANN/CNN/RNN training script
│   └── part1_2_3_4_6_7_8_9_10_combined.csv # Same dataset
│
├── neural_network_results/                 # Neural network results
│   ├── ann/                                # Artificial Neural Network results
│   │   ├── model_metrics.csv
│   │   └── predicted_vs_actual_*.png
│   ├── cnn/                                # Convolutional Neural Network results
│   │   ├── model_metrics.csv
│   │   └── predicted_vs_actual_*.png
│   ├── rnn/                                # Recurrent Neural Network results
│   │   ├── model_metrics.csv
│   │   └── predicted_vs_actual_*.png
│   ├── nn_model_metrics.csv                # NN model comparison summary
│   ├── nn_test_predictions.csv             # NN predictions on test set
│   ├── nn_model_performance_r2.png         # NN R² comparison
│   ├── nn_model_performance_rmse.png       # NN RMSE comparison
│   ├── nn_model_performance_mae.png        # NN MAE comparison
│   └── nn_model_performance_overall.png    # NN overall performance
│
├── combined_model_outputs.csv              # Combined ML + NN predictions
└── combined_model_performance.csv          # Performance comparison
```

---

## 🤖 Machine Learning Models

### Traditional ML Models (ml_training/)

| Model | Type | Description |
|-------|------|-------------|
| **XGBoost** | Gradient Boosting | Extreme Gradient Boosting with optimized hyperparameters |
| **RandomForest** | Ensemble | 500 decision trees with feature randomness |
| **GradientBoosting** | Gradient Boosting | Traditional gradient boosting with regularization |
| **Ridge Regression** | Linear | Linear regression with L2 regularization |
| **KNN** | Instance-based | K-Nearest Neighbors regression |

### Neural Network Models (nural_network/)

| Model | Type | Description |
|-------|------|-------------|
| **ANN** | Feedforward | Artificial Neural Network with dense layers |
| **CNN** | Convolutional | 1D Convolutional Neural Network |
| **RNN** | Recurrent | Recurrent Neural Network for sequential patterns |

---

## 🔬 How Prediction Works

### 1. Data Preprocessing
```
Raw Input (3000 samples, 9 features)
    ↓
Feature Scaling (StandardScaler)
    ↓
Train/Test Split (85% / 15%)
    ↓
Model Training (Cross-validation)
```

### 2. Model Training Process

#### XGBoost Configuration
- **Estimators**: 300 trees
- **Learning Rate**: 0.05 (prevents overfitting)
- **Max Depth**: 6 (tree complexity control)
- **Subsample**: 90% (row sampling)
- **Column Sample**: 90% (feature sampling)
- **Regularization**: L2 = 1.0

#### Random Forest Configuration
- **Estimators**: 500 trees
- **Max Depth**: 15 (limit overfitting)
- **Min Samples Split**: 10
- **Min Samples Leaf**: 5
- **Max Features**: √n (automatic feature selection)

#### Neural Network Architecture
- **Input Layer**: 9 neurons (one per feature)
- **Hidden Layers**: Multiple dense layers with ReLU activation
- **Dropout**: 20% (prevents overfitting)
- **Output Layer**: 4 neurons (Voc, Jsc, FF, η)
- **Optimizer**: Adam with learning rate scheduling

### 3. Prediction Workflow
```
Input: 9 Material Parameters
    ↓
[Trained Model]
    ↓
Output: 4 Performance Metrics (Voc, Jsc, FF, η)
    ↓
Validation: R² Score, RMSE, MAE
```

### 4. Model Evaluation Metrics

- **R² (Coefficient of Determination)**: Measures how well predictions match actual values (0-1, higher is better)
  - R² = 0.8 means 80% of variance is explained
- **RMSE (Root Mean Square Error)**: Average prediction error magnitude
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **Accuracy within ±15%**: Percentage of predictions within 15% of actual values

---

## 📈 Model Performance Results

### Traditional ML Models

| Model | Avg R² | Avg RMSE | Avg MAE | Best For |
|-------|--------|----------|---------|----------|
| **XGBoost** | **0.79** | **0.50** | **0.39** | Overall performance |
| **RandomForest** | 0.76 | 0.57 | 0.50 | Feature importance |
| **GradientBoosting** | 0.79 | 0.53 | 0.42 | Consistent results |
| **Ridge** | 0.75 | 0.65 | 0.57 | Interpretability |
| **KNN** | 0.68 | 0.74 | 0.62 | Simple patterns |

### Per-Target Performance (Best Model - XGBoost)

| Target | R² | RMSE | MAE | Accuracy (±15%) |
|--------|-----|------|-----|-----------------|
| **Jsc** | **0.962** | 0.53 | 0.42 | 98.2% |
| **Voc** | **0.801** | 0.036 | 0.029 | 99.8% |
| **η** | **0.890** | 0.60 | 0.45 | 96.4% |
| **FF** | **0.513** | 0.84 | 0.68 | 87.6% |

**Key Insights:**
- ✅ **Jsc**: Excellent prediction (R² = 96.2%) - Strong relationship with thickness and bandgap
- ✅ **Voc**: Good prediction (R² = 80.1%) - Predictable from bandgap differences
- ✅ **η**: Excellent prediction (R² = 89.0%) - Natural combination of Voc, Jsc, FF
- ⚠️ **FF**: Moderate prediction (R² = 51.3%) - Complex dependencies, harder to predict

### Neural Network Models

| Model | Avg R² | Avg RMSE | Avg MAE |
|-------|--------|----------|---------|
| **ANN** | 0.82 | 0.48 | 0.37 |
| **CNN** | 0.78 | 0.52 | 0.41 |
| **RNN** | 0.75 | 0.56 | 0.44 |

---

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package manager)
```

### Required Packages
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn tensorflow keras
```

### Quick Start

#### 1. Run Traditional ML Models
```bash
cd R-SolarCell/ml_training
python training.py
```

**Output**: Results saved in `ml_training/results_part1_2_3_4_6_7_8_9_10/`

#### 2. Run Neural Network Models
```bash
cd R-SolarCell/nural_network
python neural_network_training.py
```

**Output**: Results saved in `neural_network_results/`

---

## 📊 Understanding the Results

### Performance Comparison Charts

1. **model_performance_r2.png**: Bar chart comparing R² scores across all models and targets
   - Higher bars = better predictions
   - Look for bars above 0.8 (80% variance explained)

2. **model_performance_rmse.png**: Bar chart comparing RMSE across models
   - Lower bars = smaller prediction errors
   - Units match the target variable

3. **model_performance_mae.png**: Bar chart comparing MAE across models
   - Lower bars = better average accuracy

4. **model_performance_overall.png**: Combined view of all metrics

### Prediction Scatter Plots

**predicted_vs_actual_[MODEL]_[TARGET].png**
- **X-axis**: Actual values from SCAPS simulation
- **Y-axis**: Model predictions
- **Red dashed line**: Perfect prediction (y = x)
- **Green dashed lines**: ±15% error bands
- **Color gradient**: Prediction error magnitude (yellow = low error, dark = high error)
- **Metrics box**: R², RMSE, MAE, and sample count

**Interpretation**: Points closer to the red line = better predictions

---

## 🔍 Physical Insights

### Why Jsc is Easier to Predict (R² = 0.96)
- **Direct relationship** with absorber layer thickness (more material = more absorption)
- **Strong correlation** with bandgap (determines photon collection range)
- **Linear-ish behavior** in the operating range

### Why FF is Harder to Predict (R² = 0.51)
- **Complex dependencies**: Series resistance, shunt resistance, recombination losses
- **Non-linear effects**: Diode ideality factor, temperature sensitivity
- **Multiple competing mechanisms**: Not a simple function of input parameters

### Key Relationships Discovered
1. **Voc ≈ 0.6 × (Eg_L2 + Eg_L3)/2 - 0.2 × |χ_L2 - χ_L3|**
   - Increases with average bandgap
   - Decreases with electron affinity mismatch

2. **Jsc ∝ (1 - exp(-1.5 × total_thickness)) × (3.0 - avg_bandgap)**
   - Saturates at high thickness (absorption limit)
   - Inversely related to bandgap

3. **η = (Voc × Jsc × FF) / 100**
   - Natural combination of the three metrics
   - Efficiency prediction benefits from accurate Voc and Jsc

---

## 🛠️ Advanced Usage

### Hyperparameter Tuning

Edit `training.py` to modify model parameters:

```python
# XGBoost hyperparameters
XGBRegressor(
    n_estimators=500,        # More trees
    learning_rate=0.01,      # Slower learning
    max_depth=4,             # Simpler trees
    subsample=0.8,           # Row sampling
    colsample_bytree=0.8,    # Feature sampling
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=2.0,          # L2 regularization
)
```

### Adding New Features

Edit `training.py` to add engineered features:

```python
# Feature engineering examples
clean_df['thickness_ratio'] = thickness_L2 / thickness_L3
clean_df['bandgap_diff'] = bandgap_L2 - bandgap_L3
clean_df['affinity_mismatch'] = abs(affinity_L2 - affinity_L3)
```

### Using Your Own Data

Replace the CSV file with your own data (same column format):

```python
# In training.py, change:
df = pd.read_csv("your_data.csv")
```

---

## 📝 Citation & References

### Simulation Software
- **SCAPS 1-D**: Developed by Marc Burgelman, University of Ghent
- **Reference**: Burgelman, M., et al. "Modelling polycrystalline semiconductor solar cells." *Thin Solid Films* 361-362 (2000): 527-532.

### Machine Learning Libraries
- **scikit-learn**: Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python." *JMLR* 12 (2011): 2825-2830.
- **XGBoost**: Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." *KDD* (2016): 785-794.

### Solar Cell Physics
- **Perovskite Solar Cells**: Green, M. A., et al. "Organohalide lead perovskites for photovoltaic applications." *Energy & Environmental Science* 7 (2014): 2619-2623.

---

## 📄 License & Contact

**Project**: R-SolarCell - ML-based Solar Cell Performance Prediction  
**Version**: 1.0  
**Date**: April 2026  
**Author**: [Your Name/Institution]

For questions or collaborations, please contact: [your.email@institution.edu]

---

## 🎯 Future Improvements

- [ ] Add more layers (4-5 layer structures)
- [ ] Include temperature and illumination intensity as features
- [ ] Implement Bayesian optimization for hyperparameter tuning
- [ ] Add uncertainty quantification (prediction intervals)
- [ ] Deploy as web application for easy access
- [ ] Add more advanced models (Transformer, AutoML)

---

## ✅ Checklist for Reproducibility

- [x] All data preprocessing steps documented
- [x] Model hyperparameters explicitly stated
- [x] Random seed fixed (seed = 42) for reproducibility
- [x] Train/test split clearly defined (85%/15%)
- [x] Evaluation metrics explained
- [x] Code well-commented and readable
- [x] Results saved in accessible format

---

**Last Updated**: April 26, 2026
