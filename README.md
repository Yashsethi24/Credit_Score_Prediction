# CreditScore Classification: Categorizing customers' credit score into Good,Standard, or Bad 

## Project Overview
This project focuses on developing a machine learning model to predict an individual's credit score based on their financial history and personal information. Accurate credit score classification is vital for financial institutions to assess creditworthiness and make informed lending decisions.  In addition to building an ML model, we have also performed causal inference to determine the effect of certain parameters on credit scores. This analysis provides insights into which factors have a significant impact, helping customers understand how they can improve their creditworthiness.

# Problem Statement

Financial institutions need reliable methods to evaluate the credit risk associated with potential borrowers. Traditional credit scoring models may not fully capture the complexities of individual financial behaviors, leading to inaccurate assessments. This project aims to enhance credit score prediction accuracy by leveraging machine learning techniques on comprehensive datasets while also incorporating causal inference to identify key factors influencing credit scores.

# Objectives

### Data Pre-processing
Clean data, remove anomalies and make data ready for the analysis. (Refer to notebook: Data_pre_process_Credit_score_classification.ipynb) 

### Exploratory Data Analysis (EDA) and Feature Engineering
Conduct an in-depth **EDA** to identify key financial and demographic features that significantly influence credit score classification. Use **feature engineering techniques** to enhance the model performance. (Refer to notebook: EDA_Credit_score_classification_v2.ipynb) 

### Predictive modelling
Develop state-of-the-art supervised learning models (such as **XGBoost, LightGBM, CatBoost, and other ensemble techniques**) capable of accurately classifying individuals into predefined credit score categories (e.g., 'Good', 'Standard', 'Poor'). Evaluate the model's performance using cross validation accuracy metrics and validate its generalizability on unseen data.
Stacking Model:
Use **stacking ensemble approach** using different **meta-models** to enhance prediction accuracy. The base models used for stacking include:
- **LightGBM**
- **XGBoost**
- **Random Forest**

Each stacking model was trained with **Logistic Regression**.
(Refer to notebook: SupervisedLearning.ipynb) 

### Unsupervised Learning Insights
Utilize unsupervised learning techniques to cluster data and identify underlying patterns related to credit scores. (Refer to notebook: UnsupervisedLearning.ipynb) 

### Causal Inference
Conduct causal inference analysis to determine the causal impact of various factors on credit scores, offering actionable insights for customers to improve their creditworthiness.
(Refer to notebook: Causal_Inference.ipynb)

---
