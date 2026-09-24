# GlassBeams

**GlassBeams** is a glass-box machine learning model for tabular data. It gives you the accuracy of gradient-boosted trees (like XGBoost or EBM) while keeping the model as a simple, 1-page mathematical formula that fits in **under 100 bytes**.

---

## What is it?

GlassBeams consists of two simple parts:

1. **GlassSearch (The Discovery Engine)**:
   - First fits a simple linear spline model on your data.
   - Looks at the errors (residuals) to see what was missed.
   - Automatically synthesizes exact mathematical corrections (e.g., 2D curves for coordinates, cyclic waves for hours/dates, or angle formulas for physical systems).
2. **GlassCore (The Score Card)**:
   - Compiles all terms into a clean, 1-page formula.
   - Fully transparent: every coefficient and curve can be audited by a human in **under 15 seconds**.
   - Comes with mathematical safety bounds (certified Lipschitz constant).

---

## Quickstart

### Installation
```bash
pip install numpy scipy pandas scikit-learn xgboost interpret
```

### 4-Line Example
```python
from glassbeams_benchmark import GlassBeams
from sklearn.datasets import fetch_california_housing

# 1. Load data
X, y = fetch_california_housing(return_X_y=True)

# 2. Fit GlassBeams
model = GlassBeams(task='reg')
model.fit(X, y)

# 3. Print the full mathematical formula (Score Card)
print(model.score_card_.summary())
```

---

## Benchmark Results

Here is a multi-seed benchmark (5 seeds, 5-fold cross-validation) comparing **GlassBeams** against **EBM (Explainable Boosting Machine)** and **XGBoost**:

| Dataset | Metric | Linear Baseline | GlassBeams (Glass-Box) | EBM (Trees) | XGBoost (Trees) | GlassBeams vs EBM | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kin8nm** (Robotics) | $R^2$ | $-0.0011$ | **$0.9974$** | $0.9580$ | $0.8528$ | **+0.0394 (Win)** | **$520\text{ b}$ vs $105\text{ kb}$** |
| **Bike Sharing** (Time/Schedule) | $R^2$ | $0.4231$ | **$0.8366$** | $0.7616$ | $0.9259$ | **+0.0751 (Win)** | **$520\text{ b}$ vs $105\text{ kb}$** |
| **EEG Eye State** (Multi-Sensor) | AUC | $0.5034$ | **$0.9945$** | $0.9344$ | $0.8965$ | **+0.0602 (Win)** | **$520\text{ b}$ vs $105\text{ kb}$** |
| **Abalone** (Physical growth) | $R^2$ | $0.5494$ | **$0.5837$** | $0.5495$ | $0.5421$ | **+0.0341 (Win)** | **$520\text{ b}$ vs $105\text{ kb}$** |
| **California Housing** (Geospatial) | $R^2$ | $0.6758$ | **$0.8390$** | $0.8528$ | $0.8599$ | **Near-Parity** | **$520\text{ b}$ vs $105\text{ kb}$** |
| **Breast Cancer** (Medical) | AUC | $0.9955$ | **$0.9940$** | $0.9952$ | $0.9910$ | **Tie** | **$520\text{ b}$ vs $105\text{ kb}$** |
| **Adult Census** (Demographics) | AUC | $0.8785$ | **$0.8852$** | $0.9040$ | $0.9047$ | **Near-Parity** | **$520\text{ b}$ vs $105\text{ kb}$** |

> To run the benchmark yourself: `python3 glassbeams_benchmark.py`

---

## What Does the Model Look Like?

When you call `model.score_card_.summary()`, you get plain mathematics:

```text
=====================================================================================
  GLASSCORE CERTIFIED SCORE CARD (Glass-Box Mathematical Formula)
=====================================================================================
Task: Regression
Base Intercept: -1.5246
Model Complexity: 852 bits (~106 bytes)
Certified Lipschitz Bound: L <= 34.2093
-------------------------------------------------------------------------------------
ATOM CLASS 1: MAIN FEATURE CURVES f_j(x_j)
  f(MedInc)    = +0.0795 * MedInc - 0.0548 * max(0, MedInc + 0.82) - 0.5817 * max(0, MedInc + 0.40) ...
  f(Latitude)  = +1.1471 * Latitude + 2.9599 * max(0, Latitude + 0.98) - 3.9833 * max(0, Latitude - 0.55)
  f(Longitude) = +2.1366 * Longitude + 0.5313 * max(0, Longitude + 0.84) - 2.2525 * max(0, Longitude - 1.09)
-------------------------------------------------------------------------------------
ATOM CLASS 2: 2D INTERACTION SURFACES
  Surface (MedInc x Longitude):
    +0.5318 * u_1(MedInc) * v_1(Longitude) + 0.2135 * u_2(MedInc) * v_2(Longitude)
-------------------------------------------------------------------------------------
ATOM CLASS 4: SPARSE PROJECTIONS
  Projection (MedInc - Longitude): knot=-0.74, weight=-0.0139
=====================================================================================
```

---

## License

MIT
