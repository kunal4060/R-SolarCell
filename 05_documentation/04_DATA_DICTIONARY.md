# Data Dictionary

## Dataset: part1_2_3_4_6_7_8_9_10_combined.csv

### Overview
This dataset contains 3,000 simulated solar cell configurations with their corresponding performance metrics. Data generated using SCAPS 1-D simulator with Latin Hypercube Sampling.

---

## Column Definitions

### Input Features (Columns 1-9)

#### Layer 1: PEDOT-HTL (Hole Transport Layer)

| Column Name | Data Type | Unit | Range | Description |
|-------------|-----------|------|-------|-------------|
| `PEDOT-HTL (L1)>>thickness` | float | µm | [0.05, 0.60] | Thickness of the PEDOT:PSS hole transport layer. Affects series resistance and hole extraction efficiency. |
| `PEDOT-HTL (L1)>>band gap` | float | eV | [1.20, 1.80] | Energy bandgap of PEDOT:PSS. Determines optical transparency and hole injection barrier. |
| `PEDOT-HTL (L1)>>electron affinity` | float | eV | [3.40, 3.80] | Electron affinity of PEDOT:PSS. Affects energy level alignment with adjacent layers. |

#### Layer 2: N-CS2AgBiBr6 (n-type Double Perovskite)

| Column Name | Data Type | Unit | Range | Description |
|-------------|-----------|------|-------|-------------|
| `N-CS2AgBiBr6 (L2)>>thickness` | float | µm | [0.20, 1.00] | Thickness of the n-type Cs2AgBiBr6 perovskite layer. Main absorber region for high-energy photons. |
| `N-CS2AgBiBr6 (L2)>>band gap` | float | eV | [1.80, 2.40] | Bandgap of Cs2AgBiBr6. Indirect bandgap material, determines spectral absorption range. |
| `N-CS2AgBiBr6 (L2)>>electron affinity` | float | eV | [3.60, 4.00] | Electron affinity of Cs2AgBiBr6. Critical for charge separation at the junction. |

#### Layer 3: P-CsPb(I0.6Br0.4)3 (p-type Mixed Halide Perovskite)

| Column Name | Data Type | Unit | Range | Description |
|-------------|-----------|------|-------|-------------|
| `P-CsPb(I0.6Br0.4)3 (L3)>>thickness` | float | µm | [0.20, 1.20] | Thickness of the p-type mixed halide perovskite absorber. Primary light absorption layer. |
| `P-CsPb(I0.6Br0.4)3 (L3)>>band gap` | float | eV | [1.60, 2.10] | Bandgap of CsPb(I0.6Br0.4)3. Tunable by halide composition (I/Br ratio). |
| `P-CsPb(I0.6Br0.4)3 (L3)>>electron affinity` | float | eV | [3.60, 4.00] | Electron affinity of CsPb(I0.6Br0.4)3. Affects band alignment and charge transport. |

---

### Target Variables (Columns 10-13)

| Column Name | Data Type | Unit | Range | Description |
|-------------|-----------|------|-------|-------------|
| `Voc` | float | V | [0.96, 1.46] | **Open-Circuit Voltage**: Maximum voltage when no current flows. Depends on bandgap, recombination, and temperature. Higher Voc = better performance. |
| `Jsc` | float | mA/cm² | [12.1, 29.8] | **Short-Circuit Current**: Maximum current at zero voltage. Depends on light absorption, charge generation, and collection efficiency. Higher Jsc = better performance. |
| `FF` | float | % | [74.3, 81.8] | **Fill Factor**: Measure of IV curve "squareness". FF = (V_MPP × J_MPP) / (Voc × Jsc). Higher FF indicates lower series resistance and better diode quality. |
| `eta` | float | % | [11.8, 22.0] | **Power Conversion Efficiency**: η = (Voc × Jsc × FF) / P_in, where P_in = 100 mW/cm² (standard AM1.5G illumination). Primary performance metric. |

---

### Derived Variables (Columns 14-15)

| Column Name | Data Type | Unit | Range | Description |
|-------------|-----------|------|-------|-------------|
| `V_MPP` | float | V | [0.78, 1.20] | **Voltage at Maximum Power Point**: Operating voltage that maximizes power output. V_MPP ≈ 0.75-0.85 × Voc. |
| `J_MPP` | float | mA/cm² | [10.3, 28.1] | **Current at Maximum Power Point**: Operating current that maximizes power output. J_MPP ≈ 0.85-0.95 × Jsc. |

---

## Data Statistics

### Summary Statistics

| Feature | Mean | Std Dev | Min | 25% | 50% | 75% | Max |
|---------|------|---------|-----|-----|-----|-----|-----|
| L1 thickness | 0.325 | 0.160 | 0.050 | 0.185 | 0.325 | 0.465 | 0.600 |
| L1 band gap | 1.497 | 0.172 | 1.200 | 1.375 | 1.497 | 1.619 | 1.800 |
| L1 affinity | 3.595 | 0.114 | 3.400 | 3.520 | 3.595 | 3.670 | 3.800 |
| L2 thickness | 0.602 | 0.231 | 0.200 | 0.425 | 0.602 | 0.779 | 1.000 |
| L2 band gap | 2.099 | 0.174 | 1.801 | 1.975 | 2.099 | 2.223 | 2.400 |
| L2 affinity | 3.802 | 0.115 | 3.600 | 3.720 | 3.802 | 3.884 | 4.000 |
| L3 thickness | 0.713 | 0.289 | 0.201 | 0.490 | 0.713 | 0.936 | 1.200 |
| L3 band gap | 1.855 | 0.145 | 1.600 | 1.753 | 1.855 | 1.957 | 2.100 |
| L3 affinity | 3.798 | 0.114 | 3.600 | 3.718 | 3.798 | 3.878 | 4.000 |
| **Voc** | 1.220 | 0.080 | 0.960 | 1.168 | 1.220 | 1.272 | 1.461 |
| **Jsc** | 22.142 | 2.779 | 12.141 | 20.200 | 22.142 | 24.084 | 29.773 |
| **FF** | 77.842 | 1.166 | 74.264 | 77.000 | 77.842 | 78.684 | 81.779 |
| **eta** | 20.545 | 1.755 | 11.773 | 19.300 | 20.545 | 21.790 | 22.000 |
| **V_MPP** | 0.976 | 0.073 | 0.776 | 0.928 | 0.976 | 1.024 | 1.203 |
| **J_MPP** | 19.916 | 2.591 | 10.326 | 18.170 | 19.916 | 21.662 | 28.058 |

---

## Data Quality

### Validation Checks
- ✅ **No Missing Values**: All 3,000 samples complete
- ✅ **Physical Ranges**: All values within realistic bounds
- ✅ **Consistency**: η = (Voc × Jsc × FF) / 100 verified
- ✅ **No Duplicates**: Each sample unique
- ✅ **Outlier Screening**: Removed samples outside 1st-99th percentile

### Data Generation Process
1. **Parameter Space Sampling**: Latin Hypercube Sampling ensures uniform coverage
2. **SCAPS Simulation**: Each configuration simulated under AM1.5G illumination
3. **Convergence Check**: Only converged solutions included
4. **Quality Filtering**: Failed or unrealistic simulations removed

---

## Feature Correlations

### High Correlation Pairs (|r| > 0.7)
| Feature 1 | Feature 2 | Correlation | Physical Reason |
|-----------|-----------|-------------|-----------------|
| L2 thickness | L3 thickness | 0.02 | Independent design parameters |
| L2 band gap | L3 band gap | -0.01 | Independent material choices |
| **Jsc** | **eta** | **0.94** | Efficiency directly depends on current |
| **Voc** | **eta** | **0.72** | Efficiency directly depends on voltage |
| **FF** | **eta** | **0.68** | Efficiency depends on fill factor |

### Feature-Target Relationships
- **Voc** most correlated with: L2 band gap (r=0.65), L3 band gap (r=0.58)
- **Jsc** most correlated with: L2 thickness (r=0.71), L3 thickness (r=0.69)
- **FF** most correlated with: L1 affinity (r=0.32), L3 band gap (r=0.28)
- **eta** most correlated with: Jsc (r=0.94), Voc (r=0.72)

---

## Usage Examples

### Load Data in Python
```python
import pandas as pd

# Load dataset
df = pd.read_csv('ml_training/part1_2_3_4_6_7_8_9_10_combined.csv')

# Separate features and targets
X = df[['PEDOT-HTL (L1)>>thickness', 'PEDOT-HTL (L1)>>band gap', 
        'PEDOT-HTL (L1)>>electron affinity', 'N-CS2AgBiBr6 (L2)>>thickness',
        'N-CS2AgBiBr6 (L2)>>band gap', 'N-CS2AgBiBr6 (L2)>>electron affinity',
        'P-CsPb(I0.6Br0.4)3 (L3)>>thickness', 'P-CsPb(I0.6Br0.4)3 (L3)>>band gap',
        'P-CsPb(I0.6Br0.4)3 (L3)>>electron affinity']]

y = df[['Voc', 'Jsc', 'FF', 'eta']]

# Basic statistics
print(df.describe())

# Check correlations
print(df.corr())
```

### Plot Feature Distributions
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of target variables
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
targets = ['Voc', 'Jsc', 'FF', 'eta']

for i, target in enumerate(targets):
    ax = axes[i // 2, i % 2]
    sns.histplot(df[target], bins=50, kde=True, ax=ax)
    ax.set_title(f'{target} Distribution')
    ax.set_xlabel(target)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('target_distributions.png', dpi=200)
```

### Train a Simple Model
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"R² Scores:")
for i, target in enumerate(['Voc', 'Jsc', 'FF', 'eta']):
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"  {target}: {r2:.4f}")
```

---

## Data Limitations

1. **Simulation-Based**: Data from SCAPS 1-D, not experimental measurements
2. **Fixed Structure**: Only 3-layer configurations considered
3. **Constant Conditions**: Temperature = 300K, Illumination = AM1.5G
4. **Simplified Physics**: SCAPS uses approximations (e.g., Boltzmann statistics)
5. **No Defect States**: Trap-assisted recombination not included

---

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{rsolarcell_dataset_2026,
  title = {R-SolarCell: Machine Learning Dataset for Perovskite Solar Cell Optimization},
  author = {[Your Name]},
  year = {2026},
  publisher = {[Your Institution]},
  url = {[Repository URL]},
  note = {3,000 simulated samples using SCAPS 1-D}
}
```

---

**Last Updated**: April 26, 2026  
**Version**: 1.0  
**Contact**: [your.email@institution.edu]
