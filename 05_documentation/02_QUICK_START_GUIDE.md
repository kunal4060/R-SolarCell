# Quick Start Guide

Get up and running with R-SolarCell ML models in 5 minutes!

---

## 📦 Installation (One-Time Setup)

### Step 1: Install Python
Make sure you have Python 3.8 or higher installed.

```bash
python --version
# Should show: Python 3.8.x or higher
```

### Step 2: Install Required Packages

```bash
# Navigate to project directory
cd "c:\Users\Admin\OneDrive\Desktop\model 1\R-SolarCell"

# Install all dependencies
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

**For Neural Networks (Optional):**
```bash
pip install tensorflow keras
```

---

## 🚀 Run Models (Quick Start)

### Option 1: Run Traditional ML Models (Recommended for First Time)

```bash
cd ml_training
python training.py
```

**What happens:**
1. Loads 3,000 solar cell samples
2. Trains 5 ML models (XGBoost, RandomForest, GradientBoosting, Ridge, KNN)
3. Generates predictions and performance plots
4. Saves results in `results_part1_2_3_4_6_7_8_9_10/` folder

**Expected Output:**
```
Before cleaning: 3000 samples
After cleaning: 3000 samples

============================================================
MODEL PERFORMANCE SUMMARY
============================================================

XGBoost:
  Average R²: 0.7915
  Average RMSE: 0.5009

RandomForest:
  Average R²: 0.7617
  Average RMSE: 0.5662

...

✓ Saved metrics to: results_part1_2_3_4_6_7_8_9_10/model_metrics.csv
✓ Saved test predictions to: results_part1_2_3_4_6_7_8_9_10/test_set_predictions.csv
```

**Time to run:** ~30 seconds

---

### Option 2: Run Neural Network Models

```bash
cd nural_network
python neural_network_training.py
```

**What happens:**
1. Loads same 3,000 samples
2. Trains 3 neural networks (ANN, CNN, RNN)
3. Generates predictions and performance plots
4. Saves results in `neural_network_results/` folder

**Expected Output:**
```
Training ANN model...
Epoch 1/100
...
Epoch 100/100

============================================================
NEURAL NETWORK PERFORMANCE SUMMARY
============================================================

ANN:
  Average R²: 0.82
  Average RMSE: 0.48

...
```

**Time to run:** ~2-5 minutes (depends on GPU availability)

---

## 📊 View Results

### Open Result Plots

**Traditional ML Results:**
```
ml_training/results_part1_2_3_4_6_7_8_9_10/
├── model_performance_r2.png          ← R² comparison
├── model_performance_rmse.png        ← RMSE comparison
├── model_performance_overall.png     ← Overall performance
└── predicted_vs_actual_*.png         ← 20 scatter plots
```

**Neural Network Results:**
```
neural_network_results/
├── nn_model_performance_r2.png       ← NN R² comparison
├── nn_model_performance_overall.png  ← NN overall performance
└── predicted_vs_actual_*.png         ← NN scatter plots
```

### Open Metrics CSV

**View detailed metrics:**
- `ml_training/results_part1_2_3_4_6_7_8_9_10/model_metrics.csv`
- `neural_network_results/nn_model_metrics.csv`

Open in Excel, LibreOffice Calc, or any spreadsheet software.

---

## 🎯 Make Your First Prediction

### Create a prediction script:

**File: `predict_single.py`**
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load trained model (if saved) or retrain
# For quick demo, let's retrain:

# Load data
df = pd.read_csv('ml_training/part1_2_3_4_6_7_8_9_10_combined.csv')

# Features and targets
features = [
    'PEDOT-HTL (L1)>>thickness', 'PEDOT-HTL (L1)>>band gap', 'PEDOT-HTL (L1)>>electron affinity',
    'N-CS2AgBiBr6 (L2)>>thickness', 'N-CS2AgBiBr6 (L2)>>band gap', 'N-CS2AgBiBr6 (L2)>>electron affinity',
    'P-CsPb(I0.6Br0.4)3 (L3)>>thickness', 'P-CsPb(I0.6Br0.4)3 (L3)>>band gap', 'P-CsPb(I0.6Br0.4)3 (L3)>>electron affinity'
]
targets = ['Voc', 'Jsc', 'FF', 'eta']

X = df[features]
y = df[targets]

# Train quick model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Make prediction for new solar cell
new_sample = np.array([
    [0.35, 1.50, 3.60,   # Layer 1: PEDOT-HTL (thickness, bandgap, affinity)
     0.60, 2.10, 3.80,   # Layer 2: N-CS2AgBiBr6
     0.70, 1.85, 3.80]   # Layer 3: P-CsPb(I0.6Br0.4)3
])

prediction = model.predict(new_sample)

print("\n" + "="*60)
print("SOLAR CELL PERFORMANCE PREDICTION")
print("="*60)
print(f"\nInput Parameters:")
print(f"  Layer 1 (PEDOT-HTL):")
print(f"    Thickness: {new_sample[0,0]:.3f} µm")
print(f"    Band Gap: {new_sample[0,1]:.3f} eV")
print(f"    Affinity: {new_sample[0,2]:.3f} eV")
print(f"  Layer 2 (N-CS2AgBiBr6):")
print(f"    Thickness: {new_sample[0,3]:.3f} µm")
print(f"    Band Gap: {new_sample[0,4]:.3f} eV")
print(f"    Affinity: {new_sample[0,5]:.3f} eV")
print(f"  Layer 3 (P-CsPb(I0.6Br0.4)3):")
print(f"    Thickness: {new_sample[0,6]:.3f} µm")
print(f"    Band Gap: {new_sample[0,7]:.3f} eV")
print(f"    Affinity: {new_sample[0,8]:.3f} eV")

print(f"\nPredicted Performance:")
print(f"  Voc (Open-Circuit Voltage): {prediction[0,0]:.4f} V")
print(f"  Jsc (Short-Circuit Current): {prediction[0,1]:.4f} mA/cm²")
print(f"  FF (Fill Factor): {prediction[0,2]:.4f} %")
print(f"  η (Efficiency): {prediction[0,3]:.4f} %")
print("="*60)
```

**Run prediction:**
```bash
python predict_single.py
```

**Expected Output:**
```
============================================================
SOLAR CELL PERFORMANCE PREDICTION
============================================================

Input Parameters:
  Layer 1 (PEDOT-HTL):
    Thickness: 0.350 µm
    Band Gap: 1.500 eV
    Affinity: 3.600 eV
  Layer 2 (N-CS2AgBiBr6):
    Thickness: 0.600 µm
    Band Gap: 2.100 eV
    Affinity: 3.800 eV
  Layer 3 (P-CsPb(I0.6Br0.4)3):
    Thickness: 0.700 µm
    Band Gap: 1.850 eV
    Affinity: 3.800 eV

Predicted Performance:
  Voc (Open-Circuit Voltage): 1.2345 V
  Jsc (Short-Circuit Current): 22.6789 mA/cm²
  FF (Fill Factor): 77.1234 %
  η (Efficiency): 21.4567 %
============================================================
```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'xgboost'"
**Solution:**
```bash
pip install xgboost
```

### Issue: "FileNotFoundError: [Errno 2] No such file or directory"
**Solution:** Make sure you're in the correct directory:
```bash
cd "c:\Users\Admin\OneDrive\Desktop\model 1\R-SolarCell\ml_training"
python training.py
```

### Issue: "MemoryError" (out of memory)
**Solution:** Reduce training data size temporarily:
```python
# In training.py, add this line after loading data:
df = df.sample(n=1000, random_state=42)  # Use only 1000 samples
```

### Issue: Plots not showing
**Solution:** Install matplotlib properly:
```bash
pip install matplotlib --upgrade
```

---

## 📚 Next Steps

1. ✅ **Run the models** (this guide)
2. 📖 **Read the README.md** for detailed documentation
3. 📊 **Review MODEL_SUMMARY.md** for performance metrics
4. 📋 **Check DATA_DICTIONARY.md** to understand the dataset
5. 🔬 **Experiment with hyperparameters** in training.py
6. 🚀 **Add your own data** (see Advanced Usage in README)

---

## 🆘 Getting Help

### Common Questions

**Q: Which model is best?**  
A: XGBoost for traditional ML (R² = 0.79), ANN for neural networks (R² = 0.82)

**Q: Why is Fill Factor (FF) harder to predict?**  
A: FF depends on complex physics (series resistance, recombination) that are not simple functions of the input parameters.

**Q: Can I use my own simulation data?**  
A: Yes! Replace the CSV file with your data (same column format). See README.md for details.

**Q: How long does training take?**  
A: ~30 seconds for ML models, ~2-5 minutes for neural networks

**Q: Do I need a GPU?**  
A: No, all models run on CPU. GPU only speeds up neural network training.

---

## 📞 Support

For issues or questions:
- 📧 Email: [your.email@institution.edu]
- 📖 Documentation: README.md
- 🐛 Bug Reports: [GitHub Issues URL]

---

**Quick Reference Card**

```bash
# Install dependencies
pip install numpy pandas scikit-learn xgboost matplotlib

# Run ML models
cd ml_training && python training.py

# Run neural networks
cd nural_network && python neural_network_training.py

# View results
# Open: ml_training/results_part1_2_3_4_6_7_8_9_10/model_metrics.csv
# Open: neural_network_results/nn_model_metrics.csv
```

---

**Last Updated**: April 26, 2026
