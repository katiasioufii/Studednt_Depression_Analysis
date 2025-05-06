# 🧠 **DEMDI** - Depression and Mental Illness Detection and Insights Using Machine Learning

This project applies machine learning techniques to predict **depression** and analyze its relationships with various factors, including academic pressure, suicidal thoughts, family mental illness history, and more.

---

## 📁 **Dataset Overview**

- **Target Variable**: `Depression` (0 = No, 1 = Yes)
- **Features**:
  - **Demographics**: Age, Gender, City
  - **Education**: CGPA, Degree, Academic Pressure
  - **Lifestyle**: Sleep Duration, Work/Study Hours, Dietary Habits
  - **Mental Health Indicators**: Suicidal Thoughts, Family History, Financial Stress, Job/Study Satisfaction

---

## 🧹 **Data Cleaning & Preprocessing**

- **Data Removal**: Removed irrelevant or noisy values (e.g., inconsistent city names).
- **Categorical Encoding**: Converted categorical variables using `LabelEncoder`.
- **Normalization**: Normalized numerical columns using `StandardScaler`.
- **Column Removal**: Dropped unnecessary columns like `City`, `Gender`, and `Work Pressure` in the final model.

---

## 📊 **Exploratory Data Analysis (EDA)**

- **Key Insights**:
  - **24%** of depressed individuals are in **Class 12**.
  - **85.44%** of depressed individuals reported having **suicidal thoughts**.
  - **Family history of mental illness** is strongly correlated with both **depression** and **suicidal thoughts**.

- **Explored**:
  - Distribution of depression across cities, degrees, and gender.
  - Relationship between suicidal thoughts and family mental illness history.
  
---

## 🤖 **Machine Learning Models Used**

| Model                | Accuracy | F1 Score |
|----------------------|----------|----------|
| Logistic Regression  | ~84%     | 0.87     |
| SVM                  | ~84%     | 0.86     |
| Random Forest        | ~83%     | 0.86     |
| Gradient Boosting    | ~84%     | 0.86     |
| K-Nearest Neighbors  | ~81%     | 0.84     |

- **Best Performers**: Logistic Regression, Gradient Boosting, and Voting Classifier.

---

## 🧪 **Ensemble Techniques**

### ✅ **Voting Classifier**
- Used **soft voting** with base models: Logistic Regression, SVM, and Random Forest.
- **Accuracy**: ~84%, consistent across multiple runs.

---

### ✅ **Bagging**
- Tested **BaggingClassifier** using the same base models (LogReg, SVM, RF).
- **Result**: No significant improvement in performance; accuracy remained at **~84%**.

---

### ✅ **Stacking**
- Applied **StackingClassifier** with base estimators: Logistic Regression, SVM, and Random Forest.
- **Meta-model** (final estimator): Logistic Regression.
- **Result**: Accuracy remained at **~84%**, with no major improvements.

---

## 🔁 **Cross-Validation**

- **5-Fold Cross-Validation**: For better generalization.
- **Mean Accuracy**: ~84%, with minimal standard deviation.

---

## 📈 **Model Evaluation**

- **Confusion Matrix**: To evaluate classification performance.
- **ROC-AUC Curve**: To measure the model’s ability to distinguish between classes.
- **Cross-Validation**: Accuracy scores per fold to ensure consistency.
- **Feature Importance**: Visualization of feature importance from Random Forest.

---

## 📌 **Key Insights**

- **Family history of mental illness** is a strong factor in both depression and suicidal ideation.
- **85.44%** of depressed individuals report having **suicidal thoughts**.
- **24%** of people suffering from depression are in **Class 12**.
- **Data Quality**: Removing noisy city data (e.g., "Khaziabad", "Less Delhi") improved clarity.
- **Final Conclusion**: **84%** accuracy is the best achievable with the current dataset, though future improvements are possible using more advanced techniques.

---

## 🛠️ **Libraries Used**

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 🚀 **Future Work**

- **Deep Learning**: Explore **MLP** or **LSTM** (if time-series data becomes available).
- **Model Interpretability**: Use **SHAP** or **LIME** for model explainability.
- **Web App**: Deploy the model using **Flask** or **Streamlit** to make it more accessible.

---

## 📌 **Run This Project**

To run this project locally, install the required libraries and execute the main Python script:

```bash
pip install -r requirements.txt
python main.py
