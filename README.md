# Creditwise: Loan Approval Prediction System

Creditwise is a machine learning project designed to automate the loan eligibility process. By analyzing borrower profiles—including income, credit history, and employment status—the system predicts whether a loan should be approved, prioritizing financial risk mitigation.

---

## 🚀 Project Overview
This project explores the end-to-end machine learning lifecycle:

* **Dataset:** 1,000 rows and 20 features (19 inputs + 1 target).
* **Goal:** To solve two problems:
  1. High-risk customers sometimes get approved, leading to financial losses.
  2. Good customers sometimes get rejected, leading to loss of business.
* **Final Result:** While initial tests favored Naive Bayes, **Logistic Regression** emerged as the superior model after rigorous Feature Engineering.

---

## 📂 Repository Structure

* `notebooks/01_EDA_and_Preprocessing.ipynb`: Data cleaning, handling missing values with `SimpleImputer`, and visual distribution analysis.
* `notebooks/02_Model_Training_and_Comparison.ipynb`: Baseline training of Logistic Regression, KNN, and Naive Bayes.
* `notebooks/03_Feature_Engineering_Experiments.ipynb`: Advanced transformations and final model optimization.
* `data/`: Contains the raw and processed datasets.
* `models/`: Saved `.pkl` files for the final model and scaler.

---

## 📊 Key Insights from EDA
* **Class Distribution:** Analyzed the balance between approved and rejected loans to identify potential bias.
* **Outliers:** Utilized boxplots to detect anomalies in income and loan amounts.
* **Feature Correlation:** Used a correlation heatmap to identify how factors like `Credit_Score` and `Applicant_Income` impact approval rates.

---

## ⚙️ Methodology & Model Performance

### 1. Preprocessing
* **Imputation:** Numerical values filled with the **median**; categorical values filled with the **mode**.
* **Encoding:** Applied **Label Encoding** and **One-Hot Encoding** to transform categorical variables.
* **Scaling:** Used `StandardScaler` to normalize feature ranges for distance-based algorithms.

### 2. Comparison Table

| Model               | Precision| Accuracy  | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.79     | 0.87      | 0.80   | 0.80     |
| Naive Bayes         | 0.78     | 0.86      | 0.77   | 0.78     |
| KNN                 | 0.62     | 0.76      | 0.51   | 0.59     |

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Tools:** Jupyter Notebooks

---

## 📌 Conclusion
This project highlights the critical importance of **Feature Engineering**. While Naive Bayes provided an excellent initial baseline, the addition of scaled features and refined categorical encoding allowed **Logistic Regression** to achieve better generalization. The final model successfully balances the need to approve loans for qualified candidates while strictly mitigating the risk of default.
