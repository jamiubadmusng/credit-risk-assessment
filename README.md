# Credit Risk Assessment: Loan Default Prediction

## Predicting Creditworthiness Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data Source](https://img.shields.io/badge/Data-UCI%20ML%20Repository-orange.svg)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

**Author:** Jamiu Olamilekan Badmus  
**Email:** jamiubadmus001@gmail.com  
**LinkedIn:** [Jamiu Olamilekan Badmus](https://www.linkedin.com/in/jamiu-olamilekan-badmus-9276a8192/)  
**GitHub:** [jamiubadmusng](https://github.com/jamiubadmusng)  
**Website:** [sites.google.com/view/jamiu-olamilekan-badmus](https://sites.google.com/view/jamiu-olamilekan-badmus/)

---

## Executive Summary

This project develops a machine learning model to assess credit risk and predict loan defaults. Using the German Credit dataset, we build classification models that help financial institutions make informed lending decisions while managing risk exposure.

**Key Results:**
- Built and evaluated 6 classification models on 1,000 credit applicants
- Engineered risk-based features from 20 applicant attributes
- Implemented cost-sensitive classification for business optimization
- Achieved strong predictive performance with interpretable SHAP analysis

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Data Source](#data-source)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Key Findings](#key-findings)
9. [Business Recommendations](#business-recommendations)
10. [Future Work](#future-work)

---

## Problem Statement

Credit risk assessment is fundamental to banking operations:

- **$1.5 trillion** in consumer loans are issued annually in the US
- **Default rates** of 2-5% can significantly impact profitability
- **Regulatory requirements** (Basel III) mandate robust risk assessment
- **Automated scoring** enables faster, more consistent decisions

This project addresses: **How can we predict which loan applicants are likely to default, enabling risk-adjusted lending decisions?**

### Business Costs

In credit risk, different types of errors have asymmetric costs:
- **False Negative** (approving a defaulter): Loss of principal + interest
- **False Positive** (rejecting a good customer): Lost profit opportunity

Typically, the cost of a missed default is 5x the cost of a lost opportunity.

---

## Data Source

The dataset is the **German Credit Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).

### Dataset Characteristics

| Attribute | Value |
|-----------|-------|
| Source | UCI ML Repository |
| Observations | 1,000 credit applicants |
| Features | 20 (7 numerical, 13 categorical) |
| Target | Good Credit (70%) / Bad Credit (30%) |
| Time Period | Historical German bank data |

### Key Features

| Feature | Description |
|--------|-------------|
| checking_status | Status of existing checking account |
| duration | Loan duration in months |
| credit_history | Past credit behavior |
| purpose | Purpose of the loan |
| credit_amount | Loan amount requested |
| savings_status | Savings account balance |
| employment | Present employment duration |
| age | Age of applicant |
| housing | Housing situation (rent/own) |
| job | Job type and skill level |

---

## Project Structure

```
finance/
├── data/
│   ├── raw/                          # Original dataset
│   │   └── german_credit.data
│   └── processed/                    # Feature-engineered data
│       └── german_credit_processed.csv
├── docs/
│   ├── analysis_report.md            # Detailed analysis write-up
│   └── figures/                      # Visualization outputs
│       ├── target_distribution.png
│       ├── numerical_distributions.png
│       ├── categorical_default_rates.png
│       ├── correlation_matrix.png
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       ├── roc_pr_curves.png
│       ├── threshold_optimization.png
│       ├── shap_importance.png
│       ├── shap_beeswarm.png
│       └── risk_distribution.png
├── models/                           # Trained model artifacts
│   ├── credit_risk_model.joblib
│   └── preprocessor.joblib
├── notebooks/
│   └── credit_risk_assessment.ipynb  # Main analysis notebook
├── src/
│   └── predict_risk.py               # Standalone Python module
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── .gitignore                        # Git ignore file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jamiubadmusng/credit-risk-assessment.git
   cd credit-risk-assessment
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Jupyter Notebook

```bash
cd notebooks
jupyter notebook credit_risk_assessment.ipynb
```

### Using the Trained Model

```python
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load('models/credit_risk_model.joblib')
preprocessor = joblib.load('models/preprocessor.joblib')

# Prepare applicant data
applicant = pd.DataFrame({
    'checking_status': ['A11'],
    'duration': [24],
    'credit_history': ['A32'],
    # ... other features
})

# Preprocess and predict
X_processed = preprocessor.transform(applicant)
default_probability = model.predict_proba(X_processed)[:, 1]
print(f"Default Probability: {default_probability[0]:.2%}")
```

---

## Methodology

### 1. Data Preprocessing
- Decoded categorical variables for interpretability
- Converted target variable (1=Good → 0, 2=Bad → 1)
- Created derived features (credit-to-income ratio, monthly payment)

### 2. Feature Engineering

**Numerical Features:**
- Duration, credit amount, installment rate
- Age, residence duration, existing credits

**Categorical Features:**
- Checking account status, credit history
- Employment, housing, job type

**Derived Features:**
- Credit-to-income ratio
- Monthly payment estimate
- Stability indicators (employment, housing)

### 3. Model Training
- 6 classification algorithms evaluated
- 5-fold stratified cross-validation
- 80/20 train-test split

### 4. Cost-Sensitive Optimization
- Implemented asymmetric cost matrix
- Optimized classification threshold
- Balanced precision-recall for business needs

---

## Results

### Model Performance

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 |
|-------|------------|--------------|---------|
| **Logistic Regression** | **0.7682** | **0.8043** | **0.6055** |
| Random Forest | 0.7831 | 0.8037 | 0.5263 |
| LightGBM | 0.7900 | 0.7793 | 0.5766 |
| Gradient Boosting | 0.7767 | 0.7835 | 0.5636 |
| XGBoost | 0.7664 | 0.7512 | 0.5546 |
| Decision Tree | 0.6753 | 0.7090 | 0.4348 |

**Best Model: Logistic Regression** with Test ROC-AUC of **80.43%**

### Cost-Sensitive Analysis

- Default threshold (0.5): Total Cost = 151
- **Optimal threshold (0.30)**: Total Cost = 93
- **Cost reduction: 38.4%**

| Threshold | FP | FN | Recall | Total Cost |
|-----------|-----|-----|--------|------------|
| Default (0.50) | 16 | 27 | 55.0% | 151 |
| Optimal (0.30) | 38 | 11 | 81.7% | 93 |

---

## Key Findings

### 1. Checking Account Status is Critical
Applicants without a checking account or with low balance show significantly higher default rates. This is the strongest single predictor.

### 2. Loan Duration Matters
Longer loan durations correlate with higher default risk. Short-term loans (< 12 months) have lower default rates.

### 3. Credit History Drives Risk
Past payment behavior strongly predicts future defaults. Critical accounts have 2-3x higher default rates.

### 4. Cost-Sensitive Threshold Improves Business Outcomes
Optimizing the classification threshold for business costs reduces total expected loss.

---

## Business Recommendations

### Risk Segmentation Strategy

| Risk Level | Probability | Actual Default Rate | Action |
|------------|-------------|---------------------|--------|
| Very Low | < 20% | 11.0% | Auto-approve |
| Low | 20-40% | 17.1% | Standard approval |
| Medium | 40-60% | 54.5% | Enhanced review |
| High | 60-80% | 70.8% | Decline or collateral |
| Very High | > 80% | 72.7% | Decline |

### Implementation Roadmap

1. **Immediate**: Deploy model for new application scoring
2. **Short-term**: Implement tiered interest rates by risk category
3. **Long-term**: Integrate with loan management system

---

## Future Work

1. **Alternative Data**: Incorporate transaction history, social data
2. **Model Monitoring**: Implement drift detection and retraining
3. **Fairness Analysis**: Audit for demographic bias
4. **Deep Learning**: Explore neural networks for complex patterns

---

## References

1. UCI Machine Learning Repository - German Credit Dataset
2. Basel Committee on Banking Supervision - Basel III Framework
3. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaboration opportunities:

- **Email**: jamiubadmus001@gmail.com
- **LinkedIn**: [Jamiu Olamilekan Badmus](https://www.linkedin.com/in/jamiu-olamilekan-badmus-9276a8192/)
- **GitHub**: [jamiubadmusng](https://github.com/jamiubadmusng)
- **Website**: [sites.google.com/view/jamiu-olamilekan-badmus](https://sites.google.com/view/jamiu-olamilekan-badmus/)
