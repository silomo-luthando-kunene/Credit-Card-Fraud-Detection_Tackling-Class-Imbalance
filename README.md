# 🛡️ Credit Card Fraud Detection
**Binary Classification Project using Logistic Regression & Imbalanced Learning Techniques.**
</br>
## 📌 Project Overview
Recognising extreme class imbalance inherent in financial fraud datasets, this project implements a Logistic Regression ML model that aims to accurately classify credit card transactions in order to minimise future fraudulent credit card activity and minimise incorrect flagging of non-fraudulent transactions while managing the trade-off between Recall (determining the correct class for the transaction) and Precision (minimizing false alarms/incorrect transaction flagging for customers).
</br> 
## 🚀 Machine Learning Lifecycle
### 1. Exploratory Data Analysis (EDA)
*   **Imbalance Verification:** Confirmed that 99.83% of the data belongs to the majority class (Legitimate transactions).
*   **Correlation Analysis:** Discovered that anonymised PCA features **V11** and **V4** have the strongest positive correlation with fraud.
*   **Statistical Analysis:** Examined the distribution of the `Class` feature.

### 2. Data Preprocessing
*   **Feature Scaling:** Scaled the raw `Amount` and `Time` columns using `StandardScaler` to match the PCA-transformed features.
*   **Stratified Split:** Executed a 20% test split using `stratify = y` to ensure the fraud ratio remained consistent across training and testing sets.

### 3. Model Implementation
I implemented three iterations of **Logistic Regression** to compare handling strategies:
1.  **Baseline:** Standard Logistic Regression model on imbalanced data.
2.  **SMOTE:** Synthetic Minority Over-sampling Technique to balance the training set.
3.  **Class Weights:** Algorithmic adjustment using `class_weight = 'balanced'` to adjust for the minority class instances.
4.  **Evaluation:** Shifted focus to **AUPRC (Area Under Precision-Recall Curve)** and **F1-Score**, as standard accuracy is a misleading metric for this data.
## 📊 Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline LR** | 0.9991 | 0.8267 | 0.6327 | 0.7168 | 0.9574 |
| **LR + SMOTE** | 0.9738 | 0.0571 | 0.9184 | 0.1075 | 0.9697 |
| **LR + Class Weights** | 0.9746 | 0.0588 | 0.9184 | 0.1106 | 0.9714 |

## 💡 Key Insights

*   **The Accuracy Trap:** The baseline achieved 99.91% accuracy but missed nearly 37% of fraudulent transactions.
*   **Recall vs. Precision:** By using SMOTE and Class Weights, I boosted **Recall to 91.84%**, catching significantly more fraudulent transactions at the cost of higher false positives.
*   **Best Strategy:** `LR + Class Weights` proved most efficient, matching SMOTE's recall while maintaining a slightly better Precision and F1 Score without the need for synthetic data generation. A reduced count of false negatives (incorrect fraudulent transaction flagging) was achieved using the class weights method.

<div align="center">
  <img src="https://github.com/silomo-luthando-kunene/Credit-Card-Fraud-Detection_Tackling-Class-Imbalance/blob/main/Project_Files/cm_baseline.png" width="300" alt="Baseline Confusion Matrix"/>
  <img src="https://github.com/silomo-luthando-kunene/Credit-Card-Fraud-Detection_Tackling-Class-Imbalance/blob/main/Project_Files/cm_smote.png" width="300" alt="SMOTE Confusion Matrix"/>
  <img src="https://github.com/silomo-luthando-kunene/Credit-Card-Fraud-Detection_Tackling-Class-Imbalance/blob/main/Project_Files/cm_cw.png" width="300" alt="Class Weights Confusion Matrix"/>
</div>

## 🛠️ Tech Stack

*   **Language:** Python
*   **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (SMOTE), Matplotlib, Seaborn

## 🏁 Conclusion
In fraud detection, catching the "needle in the haystack" (Recall) is more important than overall model accuracy. This project proves that through proper resampling and model training and testing, we can build models that protect customers effectively and optimally flag fraudulent transactions efficiently.
</br>
**This project was completed as part of my upskilling in Machine Learning with my current focus on Classification problems**
