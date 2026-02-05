# Analysis Report: Credit Risk Assessment

## Predicting Loan Default Using Machine Learning

**Author:** Jamiu Olamilekan Badmus  
**Date:** February 2026  
**GitHub:** [jamiubadmusng](https://github.com/jamiubadmusng)  
**LinkedIn:** [Jamiu Olamilekan Badmus](https://www.linkedin.com/in/jamiu-olamilekan-badmus-9276a8192/)  
**Website:** [sites.google.com/view/jamiu-olamilekan-badmus](https://sites.google.com/view/jamiu-olamilekan-badmus/)

---

## 1. Introduction

Credit risk assessment is a cornerstone of financial services. The ability to accurately predict which borrowers are likely to default enables financial institutions to:

- Make informed lending decisions
- Price loans according to risk
- Maintain adequate capital reserves
- Comply with regulatory requirements (Basel III)

This analysis develops a machine learning model to classify credit applicants as likely to default or not, using the German Credit dataset.

### 1.1 Objectives

1. Build accurate predictive models for loan default
2. Identify key risk factors driving creditworthiness
3. Implement cost-sensitive classification
4. Provide interpretable results for regulatory compliance

### 1.2 Business Context

**Cost Asymmetry in Credit Risk:**
- False Negative (approving a defaulter): Lose principal + interest (high cost)
- False Positive (rejecting good customer): Lose profit opportunity (lower cost)

Typical cost ratio: FN:FP = 5:1

---

## 2. Data Overview

### 2.1 Dataset Description

The German Credit dataset contains 1,000 historical loan applications with 20 features and a binary outcome (good/bad credit).

| Attribute | Value |
|-----------|-------|
| Total Applicants | 1,000 |
| Good Credit (Non-Default) | 700 (70%) |
| Bad Credit (Default) | 300 (30%) |
| Numerical Features | 7 |
| Categorical Features | 13 |

### 2.2 Key Features

| Feature | Type | Description |
|---------|------|-------------|
| checking_status | Categorical | Status of checking account |
| duration | Numerical | Loan duration in months |
| credit_history | Categorical | Past credit behavior |
| purpose | Categorical | Loan purpose |
| credit_amount | Numerical | Loan amount |
| savings_status | Categorical | Savings account balance |
| employment | Categorical | Employment duration |
| age | Numerical | Applicant age |
| housing | Categorical | Housing situation |

---

## 3. Exploratory Analysis

### 3.1 Default Rate by Key Features

**Checking Account Status:**
- No checking account: Higher default risk
- Negative balance: Highest default risk
- Positive balance (>200 DM): Lowest default risk

**Credit History:**
- Critical accounts: 45%+ default rate
- All paid/no credits: 20-25% default rate

**Employment Duration:**
- Unemployed: Highest risk
- 7+ years employment: Lowest risk

### 3.2 Numerical Feature Patterns

- **Duration**: Longer loans correlate with higher default
- **Credit Amount**: Higher amounts show moderate increase in default
- **Age**: Younger applicants (<25) show higher risk

---

## 4. Feature Engineering

### 4.1 Derived Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| credit_income_ratio | credit_amount / installment_commitment | Debt burden indicator |
| monthly_payment | credit_amount / duration | Payment affordability |
| has_checking | checking_status != 'none' | Financial engagement |
| stable_employment | employment >= 4 years | Job stability |
| owns_property | property = 'real estate' | Asset security |

### 4.2 Feature Importance

Key predictive features identified:
1. Checking account status
2. Credit history
3. Loan duration
4. Credit amount
5. Employment stability

---

## 5. Model Development

### 5.1 Models Evaluated

1. Logistic Regression (baseline)
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. LightGBM

### 5.2 Evaluation Framework

- 5-fold stratified cross-validation
- Primary metric: ROC-AUC
- Secondary metrics: Precision, Recall, F1

### 5.3 Results

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 |
|-------|------------|--------------|---------|
| Logistic Regression | 0.7682 | **0.8043** | **0.6055** |
| Random Forest | 0.7831 | 0.8037 | 0.5263 |
| LightGBM | 0.7900 | 0.7793 | 0.5766 |
| Gradient Boosting | 0.7767 | 0.7835 | 0.5636 |
| XGBoost | 0.7664 | 0.7512 | 0.5546 |
| Decision Tree | 0.6753 | 0.7090 | 0.4348 |

**Best Model: Logistic Regression** with Test ROC-AUC of **80.43%**

---

## 6. Cost-Sensitive Analysis

### 6.1 Cost Matrix

| Outcome | Cost |
|---------|------|
| True Negative (Correct Approval) | 0 |
| True Positive (Correct Rejection) | 0 |
| False Positive (Lost Opportunity) | 1 |
| False Negative (Missed Default) | 5 |

### 6.2 Threshold Optimization

- Default threshold (0.5): Total Cost = 151
- **Optimal threshold (0.30)**: Total Cost = 93
- **Cost reduction: 38.4%**

**Performance Comparison:**

| Threshold | FP | FN | Precision | Recall | Total Cost |
|-----------|-----|-----|-----------|--------|------------|
| Default (0.50) | 16 | 27 | 67.4% | 55.0% | 151 |
| Optimal (0.30) | 38 | 11 | 56.3% | 81.7% | 93 |

---

## 7. Model Interpretability

### 7.1 SHAP Analysis

SHAP (SHapley Additive exPlanations) values provide:
- Global feature importance ranking
- Direction of feature effects
- Individual prediction explanations

**Top 5 Predictive Features (by SHAP importance):**
1. **Checking Status (No Account)** - Having no checking account reduces default risk
2. **Credit History (Critical)** - Critical account history significantly impacts default probability
3. **Installment Commitment** - Higher installment rates increase default risk
4. **Savings Status** - Low savings correlate with higher default
5. **Credit Amount** - Larger loan amounts increase default probability

### 7.2 Regulatory Compliance

Model interpretability supports:
- Fair lending requirements
- Model validation standards
- Adverse action explanations

---

## 8. Business Recommendations

### 8.1 Risk Segmentation

| Risk Level | Probability Range | Applicants (Test) | Actual Default Rate | Recommended Action |
|------------|-------------------|-------------------|---------------------|-------------------|
| Very Low | < 20% | 91 | 11.0% | Auto-approve |
| Low | 20-40% | 41 | 17.1% | Standard approval |
| Medium | 40-60% | 33 | 54.5% | Enhanced review |
| High | 60-80% | 24 | 70.8% | Decline or collateral |
| Very High | > 80% | 11 | 72.7% | Decline |

### 8.2 Implementation Strategy

1. **Automated Scoring**: Deploy model for real-time application scoring
2. **Tiered Pricing**: Risk-adjusted interest rates
3. **Portfolio Monitoring**: Track actual vs predicted default rates
4. **Model Governance**: Quarterly model review and retraining

### 8.3 Risk Mitigation

- Require checking accounts for medium-risk applicants
- Limit credit amounts based on risk score
- Shorten loan durations for higher-risk segments

---

## 9. Limitations

### 9.1 Data Limitations

- Historical German data (may not generalize to other markets)
- Limited sample size (1,000 observations)
- No temporal information for time-series analysis

### 9.2 Model Limitations

- Binary classification (no severity of default)
- Static features (no behavioral trends)
- Potential demographic biases requiring audit

---

## 10. Conclusions

This analysis demonstrates effective credit risk prediction using machine learning:

1. **Strong Predictive Performance**: The best model (Logistic Regression) achieves **80.43% ROC-AUC**, effectively discriminating between good and bad credit risks

2. **Key Risk Factors**: Checking account status, credit history, and loan duration are the strongest predictors of default

3. **Cost Optimization**: Using an optimal threshold of 0.30 reduces expected business costs by **38.4%** compared to the default threshold

4. **Interpretability**: SHAP analysis enables explanation of individual decisions, supporting regulatory compliance

5. **Risk Segmentation**: The model effectively segments applicants into 5 risk tiers with default rates ranging from 11% (Very Low) to 73% (Very High)

---

## References

1. UCI Machine Learning Repository - German Credit Dataset
2. Basel Committee on Banking Supervision - Basel III Framework
3. Lundberg, S. M., & Lee, S. I. (2017). SHAP Values

---

## Contact

For questions or collaboration, contact:

- **Email:** jamiubadmus001@gmail.com
- **GitHub:** [jamiubadmusng](https://github.com/jamiubadmusng)
- **LinkedIn:** [Jamiu Olamilekan Badmus](https://www.linkedin.com/in/jamiu-olamilekan-badmus-9276a8192/)
- **Website:** [sites.google.com/view/jamiu-olamilekan-badmus](https://sites.google.com/view/jamiu-olamilekan-badmus/)
