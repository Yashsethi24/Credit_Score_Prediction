# Credit Score Classification: Categorizing Customers' Credit Score into Good, Standard, or Poor

## Project Overview
This project focuses on developing a comprehensive machine learning solution to predict an individual's credit score based on their financial history and personal information. The project combines traditional supervised learning approaches with advanced techniques including ensemble methods, unsupervised clustering, and causal inference analysis to provide both predictive capabilities and actionable insights for improving creditworthiness.

Accurate credit score classification is vital for financial institutions to assess creditworthiness and make informed lending decisions. In addition to building predictive ML models, we perform causal inference to determine the effect of payment behavior on credit scores, providing insights into which factors have a significant impact and helping customers understand how they can improve their creditworthiness.

## Problem Statement

Financial institutions need reliable methods to evaluate the credit risk associated with potential borrowers. Traditional credit scoring models may not fully capture the complexities of individual financial behaviors or require extensive manual work, leading to inaccurate assessments. This project aims to enhance credit score prediction accuracy by leveraging machine learning techniques on comprehensive datasets while also incorporating causal inference to identify key factors influencing credit scores.

## Dataset Description

The project uses a comprehensive credit score dataset containing 100,000 customer records with 28 features including:
- **Demographic Information**: Age, Occupation, Name, SSN
- **Financial Metrics**: Annual Income, Monthly Inhand Salary, Outstanding Debt
- **Credit Behavior**: Number of Bank Accounts, Credit Cards, Loans, Interest Rate
- **Payment History**: Delay from Due Date, Number of Delayed Payments, Payment Behavior
- **Credit Profile**: Credit Mix, Credit Utilization Ratio, Credit History Age
- **Target Variable**: Credit Score (Good, Standard, Poor)

## Modeling Process

### 1. Data Pre-processing
- **Data Cleaning**: Remove anomalies and handle missing values using median imputation
- **Type Conversion**: Convert object columns to appropriate numeric types
- **Duplicate Handling**: Identify and remove duplicate records
- **Customer ID Mapping**: Fill missing names using customer ID mappings
- **Outlier Detection**: Handle illogical values in synthetic data

**Notebook**: `Data_pre_process_Credit_score_classification.ipynb`

### 2. Exploratory Data Analysis (EDA) and Feature Engineering
- **Comprehensive EDA**: Analyze distributions, correlations, and patterns across all features
- **Feature Engineering**: Create derived features including:
  - Income z-score and log transformations
  - Delay z-score for payment behavior
  - Cyclical encoding for month features (sin/cos transformations)
  - One-hot encoding for categorical variables
- **Feature Selection**: Implement multiple selection methods (RFE, Random Forest importance, LassoCV)
- **Data Visualization**: Generate insights through statistical plots and correlation matrices

**Notebook**: `EDA_Credit_score_classification_v2.ipynb`

### 3. Supervised Learning Models
Develop and evaluate multiple state-of-the-art supervised learning models:

**Base Models:**
- **XGBoost**: Gradient boosting with tree-based learning
- **LightGBM**: Light gradient boosting machine
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble learning
- **AdaBoost**: Adaptive boosting
- **K-Nearest Neighbors**: Distance-based classification

**Ensemble Approach:**
- **Stacking Ensemble**: Combines base models using meta-learners
  - Base Models: LightGBM, XGBoost, Random Forest
  - Meta-Model: Logistic Regression
- **Cross-Validation**: Stratified k-fold validation for robust evaluation
- **Feature Selection**: Recursive Feature Elimination (RFE) to identify top 10 features

**Performance Metrics:**
- Accuracy Score
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix Analysis

**Notebook**: `SupervisedLearning.ipynb`

### 4. Unsupervised Learning Insights
Utilize clustering techniques to identify underlying patterns:

**K-Means Clustering:**
- **Optimal K Selection**: Using silhouette score analysis (best k=2)
- **Cluster Analysis**: Compare variable means across clusters
- **PCA Visualization**: Dimensionality reduction for cluster visualization
- **Insights**: Identify distinct customer segments based on credit behavior patterns

**Key Findings:**
- Cluster 0: Younger customers (29.8 years) with higher credit utilization and delayed payments
- Cluster 1: Older customers (36.2 years) with better payment behavior and lower credit utilization

**Notebook**: `UnsupervisedLearning.ipynb`

### 5. Causal Inference Analysis
Conduct rigorous causal inference to understand the impact of payment behavior on credit scores:

**Research Questions:**
- How do delayed payments affect credit score likelihood?
- What is the causal impact of payment behavior on creditworthiness?
- How do credit utilization patterns influence credit scores?

**Methodology:**
- **Treatment Variables**: Delay_from_due_date, Num_of_Delayed_Payment, Delay_Zscore
- **Outcome Variable**: Credit Score (Good/Standard/Poor)
- **Control Variables**: Outstanding_Debt, Num_Bank_Accounts, Credit_Utilization_Ratio, etc.
- **Causal Models**: DoWhy framework with multiple estimation methods

**Key Insights:**
- Late payments have significant negative causal impact on credit scores
- Credit utilization ratio shows strong causal relationship with creditworthiness
- Outstanding debt levels moderate the effect of payment behavior

**Notebook**: `Causal_Inference.ipynb`

## Key Findings

### Feature Importance
Top predictive features identified through multiple selection methods:
1. **Outstanding_Debt** (10.7% importance)
2. **Interest_Rate** (8.8% importance) 
3. **Delay_from_due_date** (5.0% importance)
4. **Num_of_Delayed_Payment** (12.5% importance from LassoCV)
5. **Credit_Utilization_Ratio** (high causal impact)

### Model Performance
- **Ensemble Methods**: Stacking approach achieves highest accuracy
- **Feature Engineering**: Log transformations and cyclical encoding improve model performance
- **Cross-Validation**: Robust evaluation using stratified sampling

### Causal Insights
- **Payment Behavior**: Delayed payments have significant negative causal effect on credit scores
- **Credit Utilization**: Lower utilization ratios positively impact creditworthiness
- **Debt Management**: Outstanding debt levels significantly influence credit score outcomes

## Project Structure

```
Credit_Score_Prediction/
├── Datasets/
│   ├── train.csv                    # Original training data
│   ├── test.csv                     # Test data
│   ├── cleaned_trained_data.csv     # Preprocessed data
│   ├── feature_library.csv          # Feature-engineered dataset
│   └── train_cleaned.csv            # Final cleaned dataset
├── Notebooks/
│   ├── Data_pre_process_Credit_score_classification.ipynb  # Data preprocessing
│   ├── EDA_Credit_score_classification_v2.ipynb           # EDA and feature engineering
│   ├── SupervisedLearning.ipynb                           # ML model development
│   ├── UnsupervisedLearning.ipynb                         # Clustering analysis
│   └── Causal_Inference.ipynb                             # Causal inference
└── README.md
```

## Technologies Used

- **Python Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Machine Learning**: XGBoost, LightGBM, Random Forest, Gradient Boosting
- **Causal Inference**: DoWhy, CausalML, EconML
- **Data Processing**: Feature engineering, scaling, encoding
- **Visualization**: Statistical plots, correlation matrices, cluster visualizations

## Business Impact

This comprehensive analysis provides:
1. **Predictive Capabilities**: Accurate credit score classification for financial institutions
2. **Actionable Insights**: Clear understanding of factors affecting credit scores
3. **Customer Guidance**: Specific recommendations for improving creditworthiness
4. **Risk Assessment**: Better understanding of credit risk factors and their causal relationships

The combination of supervised learning, unsupervised clustering, and causal inference offers a holistic approach to credit score analysis, enabling both predictive accuracy and interpretable insights for stakeholders.

---
