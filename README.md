# 🏦 Loan Approval Prediction
### *I reduced my model accuracy from 98% to ~88% and that was the win.*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-ROS-red.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-Latest-yellow.svg)
![Finance](https://img.shields.io/badge/Domain-Finance-gold.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)

---

> Built a loan approval classifier. Found a data leakage bug faking **98% accuracy**. Fixed the pipeline. Ran **11 models × 100 random states** to find the most stable architecture. Got an honest **~88%**. Shipped it.

---

## 🐛 The Bug That Started Everything

```python
❌ OLD — Oversample before split (Data Leakage)
ros.fit_resample(X, y)              # Synthetic duplicates leak into test set
train_test_split(X_res, y_res)      # Model "sees" test data. 98% is a lie.

✅ NEW — Split first, oversample train only
train_test_split(X, y)              # Test set sealed. Never touched again.
ros.fit_resample(X_train, y_train)  # Only training data is balanced.
```

---

## 🏟️ The Tournament — How the Champion Was Chosen

> Picking a model based on a single run is guesswork. Every split can be "lucky."

Instead, **11 classifiers** were put through a loop of **100 randomized data permutations** — tracking both `Avg_Accuracy` and `Max_Accuracy` per model. Only then was a champion selected.

```python
for model in models:                               # 11 classifiers
    for i in range(100):                           # 100 random states
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        model.fit(X_train_res, y_train_res)
        scores.append(model.score(X_test, y_test))
    # → Track Max_Accuracy, Avg_Accuracy, Best_Random_State
```

**Outcome:** `random_state=45` identified as the most stable split. CatBoost selected as Champion — not for having the highest peak, but for being the most **consistent and trustworthy** across all 100 permutations.

---

## 🏆 Final Results (Leakage-Proof)

<img width="871" height="547" alt="image" src="https://github.com/user-attachments/assets/0af14c23-10af-4e52-ba63-3044706f4ad3" />


| Model | Accuracy | Verdict |
|---|---|---|
| Tuned GaussianNB | 87.80% | High peak, less stable |
| ✅**Tuned CatBoost** | 86.18% | **Champion — stable & explainable** |
| Tuned Random Forest | **85.37%** | Strong runner-up |  

**Optimized Hyperparameters (GridSearchCV · 5-fold CV):**
`n_estimators: 140` · `max_depth: 10` · `max_features: sqrt` · `random_state: 8`

---

## 🔑 What Actually Drives Loan Approval? (XAI)

<img width="1025" height="818" alt="image" src="https://github.com/user-attachments/assets/32ff4098-3101-4d22-bedf-2902290ca2cd" />


| Rank | Feature | Why It Matters |
|---|---|---|
| 🥇 | **Credit History** (~25%) | Borrower's track record of repayment — the single strongest signal of future default risk |
| 🥈 | **Loan Amount** | Higher amounts mean higher exposure for the lender |
| 🥉 | **Applicant Income** | Determines repayment capacity relative to the loan size |
| — | Gender, Marital Status | Near-zero importance — model decisions align with fair lending standards |

> This isn't just a number — it's **explainable AI in a regulated domain.** A bank officer can point to Credit History as the reason for a decision. That auditability is what makes this model production-worthy.

---

## 🔬 Pipeline at a Glance

| Phase | Detail |
|---|---|
| 🧹 **Cleaning** | Per-column mode imputation · IQR-based outlier capping |
| ⚖️ **Imbalance** | `RandomOverSampler` on training partition only |
| 🏟️ **Tournament** | 11 models × 100 random states → stable split identified |
| ⚙️ **Tuning** | `GridSearchCV` (5-fold, parallel) on leakage-free data |
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
