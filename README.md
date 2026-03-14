#  Customer Churn Prediction using Machine Learning

> Predicting telecom customer churn using an end-to-end ML pipeline — from EDA to deployment-ready model.

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/amitscode/customer-churn-prediction-using-ml)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
![Problem Type](https://img.shields.io/badge/Problem-Binary%20Classification-blue)
![Problem Type](https://img.shields.io/badge/Problem-Binary%20Classification-blue)

---

##  Table of Contents

- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [ML Workflow](#-ml-workflow)
- [Model Performance](#-model-performance)
- [Key Business Insights](#-key-business-insights)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Kaggle Notebook](#-kaggle-notebook)
- [Author](#-author)

---

##  Business Problem

Customer acquisition costs **5–7× more** than customer retention. Telecom companies lose customers daily due to:

- Pricing dissatisfaction
- Poor service quality
- Lack of personalized engagement
- Competitor offers

### This project answers:

| Question | Goal |
|----------|------|
| Which customers are most likely to churn? | Identify high-risk users early |
| What factors drive churn? | Understand root causes |
| Can ML help prevent churn? | Build a predictive retention tool |

### Business Impact:
- 🎯 Offer targeted retention deals to at-risk customers
- 💬 Prioritize customer support for high-risk users
- 💰 Optimize marketing budget allocation
- 📈 Increase Customer Lifetime Value (CLV)

---

##  Dataset

**Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset captures telecom customer profiles including demographics, billing, service subscriptions, and churn outcome.

| Attribute | Detail |
|-----------|--------|
| Total Customers | ~7,000 |
| Total Features | 20+ |
| Target Variable | `Churn` (Yes / No) |
| Problem Type | Binary Classification |

### Target Encoding:
```
Churn: Yes → 1  (churned)
       No  → 0  (retained)
```

### Feature Categories:
- **Demographics** — Gender, SeniorCitizen, Partner, Dependents
- **Account Info** — Tenure, Contract type, Payment method
- **Services** — PhoneService, InternetService, StreamingTV, etc.
- **Billing** — MonthlyCharges, TotalCharges

---

##  ML Workflow

The project follows a structured, reproducible machine learning pipeline:

```
Data Loading → EDA → Preprocessing → SMOTE → Model Training → Tuning → Evaluation
```

### 1.  Data Loading
- Import Telco dataset
- Inspect shape, dtypes, null values

### 2.  Exploratory Data Analysis (EDA)
Understand patterns that drive churn:

| Analysis | Insight |
|----------|---------|
| Churn distribution | Class imbalance identified |
| Contract type vs Churn | Month-to-month → highest churn |
| Monthly charges vs Churn | Higher charges → more churn |
| Tenure distribution | New customers churn most |
| Service usage patterns | Missing services = higher risk |

### 3.  Data Preprocessing
- Label encode categorical variables
- Handle missing/null values
- Feature type transformations
- Binary encode target (`Yes→1`, `No→0`)

### 4.  Handling Class Imbalance — SMOTE

Churn datasets are naturally imbalanced (far more retained than churned customers).

**Solution:** SMOTE *(Synthetic Minority Oversampling Technique)*

```
Without SMOTE: Model biased → predicts "No Churn" for everyone
With SMOTE:    Balanced training → learns churn patterns effectively
```

**How SMOTE works:**
1. Finds minority class samples (churned customers)
2. Generates synthetic neighbours between existing minority points
3. Balances class distribution before training

### 5.  Model Training

Three models trained and compared:

| Model | Type | Strength |
|-------|------|----------|
| Decision Tree | Tree-based | Interpretable, fast |
| Random Forest | Ensemble | Robust, handles variance |
| XGBoost | Gradient Boosting | High accuracy, handles complex patterns |

All models handle **non-linear relationships** and **mixed feature types** effectively.

### 6.  Hyperparameter Tuning — GridSearchCV

Systematic search over parameter combinations:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
```

### 7.  Model Evaluation

Metrics used:

| Metric | Why It Matters |
|--------|----------------|
| Accuracy | Overall correctness |
| ROC Curve | Visualizes trade-off between TPR and FPR |
| AUC Score | Summarizes ROC in a single value (higher = better) |

> **AUC = 1.0** → Perfect classifier  
> **AUC = 0.5** → Random guessing  
> **AUC = 0.82** → Strong discriminative power ✅

---

##  Model Performance

**Final Model: Random Forest (after tuning)**

| Metric | Score |
|--------|-------|
| Accuracy | ~78% |
| AUC Score | ~0.82 |

An **AUC of 0.82** indicates the model correctly ranks a churned customer above a non-churned customer **82% of the time** — strong performance for a business classification task.

---

##  Key Business Insights

| Factor | Observation | Action |
|--------|-------------|--------|
| 📄 Contract Type | Month-to-month → highest churn | Incentivize annual contracts |
| ⏳ Tenure | Short tenure → high churn risk | Onboarding programs for new users |
| 💵 Monthly Charges | Higher charges → more churn | Loyalty discounts for high-payers |
| 🛡️ Online Security | Absent → higher churn | Bundle security features |
| 🧑‍💻 Tech Support | Absent → higher churn | Proactive support outreach |

---

##  Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Class Balancing | imbalanced-learn (SMOTE) |
| Tuning | GridSearchCV |

---

##  Project Structure

```
Customer-Churn-Prediction-Using-Machine-Learning/
│
├── Customer_Churn_Prediction_using_ML.ipynb   # Main notebook
├── model.pkl                                   # Saved trained model
├── requirements.txt                            # Python dependencies
└── README.md                                   # Project documentation
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the notebook:
```bash
jupyter notebook Customer_Churn_Prediction_using_ML.ipynb
```

---

##  Future Improvements

| Improvement | Description |
|-------------|-------------|
| 🌐 Web App | Deploy via Streamlit for interactive predictions |
| 🔍 SHAP Explainability | Interpret individual predictions using SHAP values |
| 🔄 Real-Time API | Build a FastAPI endpoint for live churn scoring |
| 🗄️ CRM Integration | Connect predictions with Salesforce or HubSpot |
| 📊 Feature Importance | Deeper analysis of top churn drivers |

---

##  Kaggle Notebook

🔗 [View the full notebook on Kaggle](https://www.kaggle.com/code/amitscode/customer-churn-prediction-using-ml)

---

##  Author

**Amit Prajapati**  
Machine Learning & AI enthusiast focused on building practical, data-driven solutions.

---

> ⭐ If you found this project useful, consider starring the repository!
