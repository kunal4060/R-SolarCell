# Project Structure

```
R-SolarCell/
│
├── 📄 README.md                           # Main project documentation (394 lines)
│   ├── Project overview and goals
│   ├── Solar cell architecture diagram
│   ├── Dataset description (inputs & outputs)
│   ├── File structure explanation
│   ├── Model descriptions (ML & NN)
│   ├── How prediction works
│   ├── Performance results
│   ├── Installation instructions
│   ├── Usage examples
│   ├── Physical insights
│   ├── Advanced usage guide
│   ├── Citation information
│   └── Troubleshooting
│
├── 📄 QUICK_START.md                      # 5-minute quick start guide (334 lines)
│   ├── Installation steps
│   ├── Run ML models (30 seconds)
│   ├── Run neural networks (2-5 minutes)
│   ├── View results
│   ├── Make first prediction
│   ├── Troubleshooting tips
│   └── FAQ
│
├── 📄 MODEL_SUMMARY.md                    # Model performance summary (134 lines)
│   ├── Quick overview table
│   ├── Per-target R² scores
│   ├── Model comparison
│   ├── NN performance
│   ├── Physical relationships
│   ├── Quality assessment
│   └── Recommendations
│
├── 📄 DATA_DICTIONARY.md                  # Dataset documentation (218 lines)
│   ├── Column definitions
│   ├── Feature descriptions (9 inputs)
│   ├── Target descriptions (4 outputs)
│   ├── Derived variables
│   ├── Summary statistics
│   ├── Data quality checks
│   ├── Feature correlations
│   ├── Usage examples (Python code)
│   └── Limitations
│
├── 📄 PROJECT_STRUCTURE.md                # This file (file tree)
│
├── 📊 combined_model_outputs.csv          # Combined predictions from all models
│   ├── ML model predictions (XGBoost, RF, GB, Ridge, KNN)
│   ├── NN model predictions (ANN, CNN, RNN)
│   └── Ensemble predictions
│
├── 📊 combined_model_performance.csv      # Performance comparison
│   ├── All models' metrics
│   ├── Side-by-side comparison
│   └── Best model identification
│
│
├── 📂 ml_training/                        # Traditional Machine Learning
│   │
│   ├── 📄 training.py                     # Main training script (289 lines)
│   │   ├── Data loading and preprocessing
│   │   ├── Feature engineering
│   │   ├── Outlier removal
│   │   ├── Model definitions:
│   │   │   ├── XGBoost (300 trees, lr=0.05)
│   │   │   ├── RandomForest (500 trees)
│   │   │   ├── GradientBoosting (regularized)
│   │   │   ├── Ridge (L2 regularization)
│   │   │   └── KNN (instance-based)
│   │   ├── Model training loop
│   │   ├── Prediction generation
│   │   ├── Metric calculation (R², RMSE, MAE)
│   │   ├── Plot generation (20+ plots)
│   │   └── Results saving
│   │
│   ├── 📄 part1_2_3_4_6_7_8_9_10_combined.csv  # Training dataset
│   │   ├── 3,000 samples (rows)
│   │   ├── 15 columns (9 features + 4 targets + 2 derived)
│   │   ├── Size: 473.2 KB
│   │   └── Format: CSV (comma-separated)
│   │
│   └── 📂 results_part1_2_3_4_6_7_8_9_10/      # Output directory (37 files)
│       │
│       ├── 📊 model_metrics.csv                  # Performance metrics table
│       │   └── Columns: model, target, R2, RMSE, MAE
│       │
│       ├── 📊 test_set_predictions.csv           # Predictions on test set
│       │   ├── Actual values (y_test)
│       │   └── Predicted values (y_pred) for each model
│       │
│       ├── 📈 model_performance_r2.png           # R² bar chart
│       │   └── Compare R² across models & targets
│       │
│       ├── 📈 model_performance_rmse.png         # RMSE bar chart
│       │   └── Compare RMSE across models & targets
│       │
│       ├── 📈 model_performance_mae.png          # MAE bar chart
│       │   └── Compare MAE across models & targets
│       │
│       ├── 📈 model_performance_overall.png      # Overall performance
│       │   └── Combined metrics comparison
│       │
│       └── 📈 predicted_vs_actual_*.png          # Scatter plots (20 total)
│           ├── predicted_vs_actual_XGBoost_Voc.png
│           ├── predicted_vs_actual_XGBoost_Jsc.png
│           ├── predicted_vs_actual_XGBoost_FF.png
│           ├── predicted_vs_actual_XGBoost_eta.png
│           ├── predicted_vs_actual_RandomForest_Voc.png
│           ├── predicted_vs_actual_RandomForest_Jsc.png
│           ├── predicted_vs_actual_RandomForest_FF.png
│           ├── predicted_vs_actual_RandomForest_eta.png
│           ├── predicted_vs_actual_GradientBoosting_Voc.png
│           ├── predicted_vs_actual_GradientBoosting_Jsc.png
│           ├── predicted_vs_actual_GradientBoosting_FF.png
│           ├── predicted_vs_actual_GradientBoosting_eta.png
│           ├── predicted_vs_actual_Ridge_Voc.png
│           ├── predicted_vs_actual_Ridge_Jsc.png
│           ├── predicted_vs_actual_Ridge_FF.png
│           ├── predicted_vs_actual_Ridge_eta.png
│           ├── predicted_vs_actual_KNN_Voc.png
│           ├── predicted_vs_actual_KNN_Jsc.png
│           ├── predicted_vs_actual_KNN_FF.png
│           └── predicted_vs_actual_KNN_eta.png
│
│
├── 📂 nural_network/                      # Neural Network Training
│   │
│   ├── 📄 neural_network_training.py      # NN training script (21.5 KB)
│   │   ├── Data preprocessing
│   │   ├── ANN model definition
│   │   │   ├── Dense layers
│   │   │   ├── Dropout (20%)
│   │   │   └── Adam optimizer
│   │   ├── CNN model definition
│   │   │   ├── 1D convolutional layers
│   │   │   ├── Max pooling
│   │   │   └── Fully connected layers
│   │   ├── RNN model definition
│   │   │   ├── LSTM/GRU layers
│   │   │   └── Dense output layers
│   │   ├── Training loops
│   │   ├── Early stopping
│   │   ├── Model checkpointing
│   │   └── Prediction generation
│   │
│   └── 📄 part1_2_3_4_6_7_8_9_10_combined.csv  # Same dataset
│       └── (Symbolic link or copy of ml_training data)
│
│
└── 📂 neural_network_results/             # Neural Network Output
    │
    ├── 📂 ann/                             # ANN results
    │   ├── 📊 model_metrics.csv
    │   └── 📈 predicted_vs_actual_*.png (4 plots)
    │
    ├── 📂 cnn/                             # CNN results
    │   ├── 📊 model_metrics.csv
    │   └── 📈 predicted_vs_actual_*.png (4 plots)
    │
    ├── 📂 rnn/                             # RNN results
    │   ├── 📊 model_metrics.csv
    │   └── 📈 predicted_vs_actual_*.png (4 plots)
    │
    ├── 📊 nn_model_metrics.csv             # NN summary metrics
    │   └── Comparison of ANN, CNN, RNN
    │
    ├── 📊 nn_test_predictions.csv          # NN test predictions
    │
    ├── 📈 nn_model_performance_accuracy.png # Accuracy comparison
    ├── 📈 nn_model_performance_mae.png      # MAE comparison
    ├── 📈 nn_model_performance_overall.png  # Overall NN performance
    ├── 📈 nn_model_performance_r2.png       # R² comparison
    └── 📈 nn_model_performance_rmse.png     # RMSE comparison
```

---

## File Sizes

| File | Size | Description |
|------|------|-------------|
| `part1_2_3_4_6_7_8_9_10_combined.csv` | 473.2 KB | Main dataset (3000×15) |
| `training.py` | 9.7 KB | ML training script |
| `neural_network_training.py` | 21.5 KB | NN training script |
| `model_metrics.csv` | 4.2 KB | Performance metrics |
| `test_set_predictions.csv` | ~100 KB | Predictions |
| Each plot (PNG) | 45-90 KB | High-resolution (200 DPI) |
| **Total Project** | **~10 MB** | All files included |

---

## File Counts by Directory

| Directory | Files | Subdirectories | Total Size |
|-----------|-------|----------------|------------|
| `R-SolarCell/` (root) | 6 | 3 | ~10 MB |
| `ml_training/` | 2 | 1 | ~600 KB |
| `ml_training/results_*/` | 37 | 0 | ~2 MB |
| `nural_network/` | 2 | 0 | ~500 KB |
| `neural_network_results/` | 7 | 3 | ~5 MB |

---

## Key Files Quick Reference

### 🚀 To Run Models
- **ML Models**: `ml_training/training.py`
- **Neural Networks**: `nural_network/neural_network_training.py`

### 📊 To View Data
- **Dataset**: `ml_training/part1_2_3_4_6_7_8_9_10_combined.csv`

### 📈 To View Results
- **ML Metrics**: `ml_training/results_part1_2_3_4_6_7_8_9_10/model_metrics.csv`
- **NN Metrics**: `neural_network_results/nn_model_metrics.csv`
- **Best Plots**: `ml_training/results_*/model_performance_overall.png`

### 📖 To Read Documentation
- **Start Here**: `QUICK_START.md`
- **Full Docs**: `README.md`
- **Data Info**: `DATA_DICTIONARY.md`
- **Performance**: `MODEL_SUMMARY.md`

---

## Naming Conventions

### Dataset Files
- `part1_2_3_4_6_7_8_9_10_combined.csv`: Combined simulation batches
  - `part1`, `part2`, etc.: Different SCAPS simulation runs
  - `combined`: Merged into single dataset

### Result Files
- `model_metrics.csv`: Performance metrics for all models
- `test_set_predictions.csv`: Model predictions on test data
- `model_performance_[metric].png`: Bar charts comparing models
- `predicted_vs_actual_[MODEL]_[TARGET].png`: Scatter plots

### Model Names in Files
- `XGBoost`: Extreme Gradient Boosting
- `RandomForest`: Random Forest
- `GradientBoosting`: Traditional Gradient Boosting
- `Ridge`: Ridge Regression
- `KNN`: K-Nearest Neighbors
- `ann`: Artificial Neural Network
- `cnn`: Convolutional Neural Network
- `rnn`: Recurrent Neural Network

---

## Directory Creation

If results directories don't exist, they are created automatically by training scripts:

```python
# In training.py
output_dir = script_dir / "results_part1_2_3_4_6_7_8_9_10"
output_dir.mkdir(exist_ok=True)
```

---

**Last Updated**: April 26, 2026
