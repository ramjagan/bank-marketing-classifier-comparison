# Bank Marketing Classifier Comparison

## Project Overview
This project compares the performance of four machine learning classifiers — **K-Nearest Neighbors (KNN)**, **Logistic Regression**, **Decision Tree**, and **Support Vector Machine (SVM)** — to predict whether a client will subscribe to a term deposit based on data from a Portuguese bank's telemarketing campaigns.

**Dataset:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)  
**Notebook:** [prompt_III.ipynb](prompt_III.ipynb)

---

## Business Objective
The bank wants to reduce wasted telemarketing calls by identifying in advance which customers are most likely to subscribe to a term deposit. A predictive model helps the bank prioritize its call list, improve conversion rates, and reduce campaign costs.

---

## Dataset
- **Source:** UCI Machine Learning Repository — Bank Marketing Dataset
- **File used:** `data/bank-additional/bank-additional-full.csv`
- **Records:** 41,188 rows × 20 input features + 1 binary target (`y`)
- **Time period:** May 2008 – November 2010, Portuguese retail bank
- **Target variable:** `y` — Has the client subscribed to a term deposit? (yes/no)

---

## Key Steps
1. Exploratory Data Analysis (EDA) with visualizations
2. Feature engineering — dropped `duration` to avoid target leakage, applied `StandardScaler` and `OneHotEncoder`
3. Train/test split (80/20, stratified)
4. Baseline model — majority class accuracy: **88.7%**
5. Model comparison — Logistic Regression, KNN, Decision Tree, SVM
6. Hyperparameter tuning via `GridSearchCV` (ROC-AUC scoring)

---

## Model Comparison Results

| Model | Train Time (s) | Train Accuracy | Test Accuracy |
|---|---|---|---|
| Logistic Regression | 1.85 | 0.900 | 0.901 |
| KNN | 0.16 | 0.912 | 0.897 |
| Decision Tree | 0.44 | 0.995 | 0.842 |
| SVM (RBF) | 791.68 | 0.905 | 0.903 |

---

## Tuned Model Results

| Model | Test Accuracy | ROC-AUC | Recall (subscribers) |
|---|---|---|---|
| Tuned Logistic Regression (C=1, balanced) | 0.835 | 0.801 | **0.65** |
| Tuned Decision Tree (max_depth=5) | 0.903 | 0.791 | 0.26 |

---

## Findings & Recommendations
- The **Decision Tree overfits** significantly (train 99.5% vs test 84.2%) without tuning.
- **SVM (RBF)** achieves the highest raw test accuracy (90.3%) but takes **791 seconds** to train — impractical for large datasets.
- The **Tuned Logistic Regression** with `class_weight=balanced` is the recommended model. While its overall accuracy (83.5%) is slightly lower, it achieves a **recall of 65%** for subscribers — meaning it identifies nearly two-thirds of likely customers, greatly improving campaign targeting.

### Actionable Recommendations
1. Deploy the tuned logistic regression to **rank and prioritize call lists** by predicted subscription probability.
2. Use recall as the primary metric — catching subscribers matters more than overall accuracy in this imbalanced dataset.
3. Collect richer features (e.g., customer tenure, product history) to further improve recall.
4. Run A/B tests to measure campaign ROI improvement vs. random calling.

---

## Repository Structure
