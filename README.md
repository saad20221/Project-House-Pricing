# 🏠 House Price Prediction Intervals — ML & DevOps Pipeline

## 📌 Project Overview

This project builds a **reproducible machine learning pipeline** to predict **prediction intervals** (lower and upper bounds) for house sale prices using structured property and location features.

Instead of predicting a single price, the system produces an **80% prediction interval** per property using **quantile regression models**.

Two gradient boosting frameworks are implemented and compared:

* LightGBM (quantile objective)
* XGBoost (quantile objective)

The pipeline is fully automated and DevOps-ready: preprocessing, training, evaluation, and artifact generation are executed through a single pipeline script.

---

## 🎯 Objectives

* Build an objective, data-driven interval prediction model
* Produce lower and upper bounds for sale_price
* Ensure full reproducibility
* Compare multiple interval model families
* Report interval quality metrics:

  * Coverage
  * Interval Width (Sharpness)
  * Pinball Loss
  * Midpoint MAE
* Generate submission artifacts automatically

---

## 📂 Repository Structure

```
house-price-interval/
│
├── data/
│   │──  dataset.csv
│   │    test.csv
│
├── Pipeline.py   # Main pipeline entrypoint
│
├── artifacts/
│   ├── submission_LightGBM.csv
│   └── submission_XGBoost.csv
│
├── tests/
│   └── test_features.py
│
├── requirements.txt
├── azure-pipelines.yml
└── README.md
```

---

## ⚙️ Pipeline Workflow

The automated pipeline performs:

1. Data loading (train + test)
2. Feature engineering
3. Categorical encoding

   * Target encoding
   * Ordinal encoding
4. Missing value imputation
5. Train / validation split
6. Train interval models:

   * Quantile 0.1 model
   * Quantile 0.9 model
7. Validation interval metrics computation
8. Refit on full training data
9. Test set interval prediction
10. Artifact export

Run with:

```bash
python Pipeline.py
```

---

## 🧠 Modeling Approach

Prediction intervals are produced using **quantile regression**:

* Lower bound → 10th percentile model
* Upper bound → 90th percentile model

Interval = middle 80% of conditional price distribution.

Models used:

* LightGBM — quantile objective
* XGBoost — quantile objective

Both models use identical preprocessing to ensure fair comparison.

---

## 📊 Evaluation Metrics

### Coverage

Percentage of true prices falling inside predicted interval.

Target ≈ 0.80 for a 10%–90% interval.

### Interval Width (Sharpness)

Average size of predicted interval.

Smaller width = more precise intervals (if coverage maintained).

### Pinball Loss

Quantile loss used to evaluate bound placement quality.

Computed separately for:

* lower quantile model
* upper quantile model

### Midpoint MAE

MAE between true price and interval midpoint.
Used as secondary central accuracy indicator.

---

## 📦 Outputs

Pipeline produces:

```
artifacts/
 ├── submission_LightGBM.csv
 ├── submission_XGBoost.csv
 └── metrics.json
```

Submission format:

```
id, pi_lower, pi_upper
```

---

## 🔁 Reproducibility

* Fixed random seeds
* Deterministic preprocessing
* Encoders fit on training data only
* Same transforms applied to test data
* Models and preprocessors serialized
* CI pipeline executes full workflow

---

## 🚀 DevOps Integration

Azure DevOps pipeline performs:

* Dependency installation
* Lint checks
* Unit tests
* Model training
* Metric computation
* Artifact publishing

Run automatically on main branch updates.

---

## 🧪 Testing

Basic unit tests validate:

* Feature engineering outputs
* Transformer stability
* Column presence after transforms

Run with:

```bash
pytest
```

---

## 📚 Technical References

* Quantile Regression — Koenker
* Forecasting Prediction Intervals — Hyndman
* LightGBM Quantile Objective Docs
* XGBoost Quantile Regression Docs

---

## ✅ Status

✔ Reproducible pipeline
✔ Dual-model interval prediction
✔ DevOps automation
✔ Interval metrics reporting
✔ Artifact generation
✔ Ready for submission and review

---
