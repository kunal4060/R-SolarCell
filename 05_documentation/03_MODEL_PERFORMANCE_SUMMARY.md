# Model Performance Summary

## Quick Overview

| Metric | Value |
|--------|-------|
| **Total Samples** | 3,000 |
| **Input Features** | 9 (layer parameters) |
| **Target Variables** | 4 (Voc, Jsc, FF, η) |
| **Train/Test Split** | 85% / 15% (2,550 / 450) |
| **Best Model** | XGBoost |
| **Best Average R²** | 0.79 (79%) |

---

## Individual Target Performance

### Open-Circuit Voltage (Voc)
- **Best Model**: XGBoost
- **R² Score**: 0.801 (80.1%)
- **RMSE**: 0.036 V
- **MAE**: 0.029 V
- **Prediction Accuracy**: 99.8% within ±15%
- **Physical Range**: 0.96 - 1.46 V

### Short-Circuit Current (Jsc)
- **Best Model**: XGBoost
- **R² Score**: 0.962 (96.2%) ⭐ **EXCELLENT**
- **RMSE**: 0.53 mA/cm²
- **MAE**: 0.42 mA/cm²
- **Prediction Accuracy**: 98.2% within ±15%
- **Physical Range**: 12.1 - 29.8 mA/cm²

### Fill Factor (FF)
- **Best Model**: XGBoost
- **R² Score**: 0.513 (51.3%)
- **RMSE**: 0.84 %
- **MAE**: 0.68 %
- **Prediction Accuracy**: 87.6% within ±15%
- **Physical Range**: 74.3 - 81.8 %
- **Note**: FF has complex dependencies, making it harder to predict

### Efficiency (η)
- **Best Model**: XGBoost
- **R² Score**: 0.890 (89.0%) ⭐ **EXCELLENT**
- **RMSE**: 0.60 %
- **MAE**: 0.45 %
- **Prediction Accuracy**: 96.4% within ±15%
- **Physical Range**: 11.8 - 22.0 %

---

## Model Comparison (Average R²)

| Rank | Model | Avg R² | Best For |
|------|-------|--------|----------|
| 1 | XGBoost | 0.79 | Overall performance |
| 2 | GradientBoosting | 0.79 | Consistency |
| 3 | RandomForest | 0.76 | Feature importance |
| 4 | Ridge | 0.75 | Interpretability |
| 5 | KNN | 0.68 | Simple patterns |

---

## Neural Network Performance

| Model | Avg R² | Avg RMSE | Avg MAE |
|-------|--------|----------|---------|
| **ANN** | 0.82 | 0.48 | 0.37 |
| **CNN** | 0.78 | 0.52 | 0.41 |
| **RNN** | 0.75 | 0.56 | 0.44 |

**Winner**: Artificial Neural Network (ANN) with R² = 0.82

---

## Key Physical Relationships Discovered

1. **Voc ≈ f(bandgap_avg, affinity_mismatch)**
   - Increases with average bandgap
   - Decreases with electron affinity differences

2. **Jsc ≈ f(total_thickness, bandgap_avg)**
   - Increases with absorber thickness (up to saturation)
   - Decreases with larger bandgap

3. **η = (Voc × Jsc × FF) / 100**
   - Natural combination of all three metrics
   - Benefits from accurate Voc and Jsc predictions

---

## Prediction Quality Assessment

| Quality Level | R² Range | Targets Achieving This |
|---------------|----------|------------------------|
| **Excellent** | > 0.90 | Jsc (0.962) |
| **Good** | 0.80 - 0.90 | Voc (0.801), η (0.890) |
| **Moderate** | 0.50 - 0.80 | FF (0.513) |
| **Poor** | < 0.50 | None |

---

## Recommendations

### For Research Use
- **Use XGBoost** for Voc, Jsc, and η predictions (high accuracy)
- **Use ANN** for overall best performance across all targets
- **Use FF predictions with caution** (moderate accuracy, complex physics)

### For Production Use
- **XGBoost**: Fast inference, good accuracy, easy to deploy
- **ANN**: Better accuracy but requires more computational resources

### For Further Improvement
- Collect more simulation data (> 5,000 samples)
- Add temperature and illumination as input features
- Implement ensemble methods (combine multiple models)
- Use transfer learning with experimental data

---

## Data Quality

- ✅ All 3,000 samples validated
- ✅ No missing values
- ✅ Physically realistic ranges enforced
- ✅ Latin Hypercube Sampling for good coverage
- ✅ Outliers removed (1st-99th percentile)

---

**Last Updated**: April 26, 2026
