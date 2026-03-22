# 🏦 Loan Approval Prediction
### *I reduced my model accuracy from 98% to ~88% — and that was the win.*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-ROS-red.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-Latest-yellow.svg)
![Finance](https://img.shields.io/badge/Domain-Finance-gold.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)

---

> Built a loan approval classifier. Found a data leakage bug faking **98% accuracy**. Fixed the pipeline. Got an honest **~88%**. Shipped it. That drop is the entire story.

---

## 🐛 The Bug That Started Everything

```python
# ❌ OLD — Oversampling before splitting (Data Leakage)
ros.fit_resample(X, y)              # Synthetic duplicates leak into test set
train_test_split(X_res, y_res)      # Model "sees" test data. 98% is a lie.

# ✅ NEW — Split first, oversample train only
train_test_split(X, y)              # Test set sealed. Never touched again.
ros.fit_resample(X_train, y_train)  # Only training data is balanced.
```

---

## 🏆 Results

![Model Comparison](images/model_comparison.png)

| Model | Accuracy | Verdict |
|---|---|---|
| GaussianNB | 87.80% | High peak, less stable |
| Tuned CatBoost | 86.18% | Strong runner-up |
| ✅ **Tuned Random Forest** | **85.37%** | **Champion — stable & explainable** |

> Random Forest wins not on peak accuracy, but on **reliability across 100 random states** and **interpretability** — both non-negotiable in financial lending.

---

## 🔑 What Actually Drives Loan Approval?

![Feature Importance](images/feature_importance.png)

**Credit History alone accounts for ~25% of the decision.** Income and loan amount follow. Demographics like gender barely register — as it should be.

---

## 🔬 How It Was Built

| Phase | What Happened |
|---|---|
| 🧹 **Data Cleaning** | Per-column mode imputation for categoricals; IQR capping for outliers |
| ⚖️ **Imbalance Fix** | `RandomOverSampler` applied **strictly on training data** |
| 🏟️ **The Tournament** | 11 models × 100 random states → identified `random_state=8` as most stable |
| ⚙️ **Tuning** | `GridSearchCV` (5-fold CV, parallel) on leakage-free data |
| 📊 **Evaluation** | Confusion Matrix · Classification Report · Feature Importance |

---

## 🛠️ Stack

`Pandas` · `NumPy` · `Scikit-Learn` · `imbalanced-learn` · `CatBoost` · `XGBoost` · `Matplotlib` · `Seaborn`

---

## 🚀 Run It

```bash
git clone https://github.com/yourusername/loan-approval-prediction.git
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn catboost xgboost
# Open Loan_Approval_Prediction.ipynb
```

---

<div align="center">

*The model that admits its flaws is the one you can trust in production.*

</div>
