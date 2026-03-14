Customer Churn Prediction using Machine Learning

This project builds a machine learning system to predict telecom customer churn using the Telco Customer Churn dataset.

Customer churn occurs when a customer stops using a company's service. Predicting churn helps companies take proactive retention actions, reducing revenue loss and improving customer lifetime value.

The project demonstrates an end-to-end machine learning workflow, from data exploration to model evaluation and prediction.

Kaggle Notebook:
https://www.kaggle.com/code/amitscode/customer-churn-prediction-using-ml

Business Problem

Customer acquisition is significantly more expensive than customer retention.
Telecommunication companies often lose customers due to pricing, contract terms, service quality, or lack of engagement.

This project aims to answer the following business questions:

Which customers are most likely to churn?

What factors contribute most to churn behavior?

Can machine learning help identify high-risk customers early?

By predicting churn probability, companies can:

Offer targeted retention offers

Improve customer support for high-risk users

Optimize marketing spending

Increase customer lifetime value

Dataset

Dataset: Telco Customer Churn Dataset

The dataset contains telecom customer information including:

Customer demographics

Account information

Service subscriptions

Billing details

Churn status

Dataset characteristics:

Feature	Description
Customers	~7,000
Features	20+
Target Variable	Churn (Yes / No)
Problem Type	Binary Classification

Target Variable:

Churn

Encoded as:

Yes → 1
No → 0
Machine Learning Workflow

The project follows a structured machine learning pipeline.

1. Data Loading

The Telco dataset is imported and inspected for structure and missing values.

2. Exploratory Data Analysis (EDA)

EDA is used to understand customer behavior patterns.

Key analysis includes:

Churn distribution

Contract type vs churn

Monthly charges vs churn

Tenure distribution

Service usage patterns

This step helps identify important churn drivers.

3. Data Preprocessing

Data preparation includes:

Label encoding categorical variables

Handling missing values

Feature transformation

Converting churn labels to binary format

Example:

Yes → 1
No → 0
4. Handling Class Imbalance

The churn dataset is imbalanced (fewer churn customers than non-churn).

To address this, the project uses SMOTE (Synthetic Minority Oversampling Technique).

SMOTE generates synthetic samples for the minority class to improve model learning.

Benefits:

Reduces bias toward majority class

Improves churn detection performance

5. Model Training

Multiple machine learning models were trained and evaluated:

Decision Tree

Random Forest

XGBoost

These models were selected because they handle non-linear relationships and mixed feature types effectively.

6. Hyperparameter Tuning

Hyperparameter optimization was performed using:

GridSearchCV

This improves model performance by finding optimal values for parameters such as:

number of estimators

tree depth

7. Model Evaluation

Model performance was evaluated using:

Accuracy

ROC Curve

AUC Score

The ROC curve measures the model's ability to distinguish between churn and non-churn customers.

Model Performance

Final model: Random Forest

Performance metrics:

Metric	Score
Accuracy	~78%
AUC Score	~0.82

An AUC score of 0.82 indicates strong ability to distinguish churn vs non-churn customers.

Key Business Insights

Analysis of the dataset reveals several important churn patterns:

1. Contract Type

Customers with month-to-month contracts have significantly higher churn rates compared to long-term contracts.

2. Tenure

Customers with short tenure are more likely to churn.

3. Monthly Charges

Higher monthly charges correlate with increased churn probability.

4. Service Features

Customers lacking additional services such as:

Online security

Tech support

show higher churn risk.

These insights can help telecom companies design targeted retention strategies.

Technologies Used

Programming Language

Python

Data Processing

Pandas

NumPy

Machine Learning

Scikit-Learn

Random Forest

Decision Tree

XGBoost

Visualization

Matplotlib

Seaborn

Class Imbalance Handling

SMOTE

Project Structure
Customer-Churn-Prediction-Using-Machine-Learning
│
├── Customer_Churn_Prediction_using_ML.ipynb
├── README.md
├── model.pkl
└── requirements.txt
Future Improvements

Possible extensions to improve the project:

Deploy the model as a web application using Streamlit

Implement SHAP explainability to interpret predictions

Build a real-time churn prediction API

Integrate with customer relationship management (CRM) systems

Perform feature importance analysis for deeper business insights

Why This Project Matters

This project demonstrates practical skills required in real-world machine learning roles:

Business problem understanding

Data exploration and cleaning

Handling imbalanced datasets

Model training and evaluation

Interpreting results for business impact

These skills are critical for data science and machine learning engineering roles.

Author

Amit Prajapati

Machine Learning and AI enthusiast focused on building practical data-driven solutions.
