"""
Credit Risk Prediction Module
============================

This module provides functionality to predict credit default risk
using a pre-trained machine learning model.

Author: Jamiu Olamilekan Badmus
Email: jamiubadmus001@gmail.com
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'credit_risk_model.joblib')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'preprocessor.joblib')

# Default and optimal thresholds
DEFAULT_THRESHOLD = 0.5
OPTIMAL_THRESHOLD = 0.30

# Risk segment boundaries
RISK_SEGMENTS = {
    'Very Low': (0, 0.2),
    'Low': (0.2, 0.4),
    'Medium': (0.4, 0.6),
    'High': (0.6, 0.8),
    'Very High': (0.8, 1.0)
}

# Feature columns for the model
FEATURE_COLUMNS = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_status', 'employment', 'installment_commitment', 'personal_status',
    'other_parties', 'residence_since', 'property_magnitude', 'age',
    'other_payment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'own_telephone', 'foreign_worker'
]


class CreditRiskPredictor:
    """
    Credit Risk Prediction class that loads a pre-trained model
    and provides methods for predicting default probabilities.
    """
    
    def __init__(self, model_path: str = MODEL_PATH, 
                 preprocessor_path: str = PREPROCESSOR_PATH):
        """
        Initialize the predictor with model and preprocessor.
        
        Parameters
        ----------
        model_path : str
            Path to the trained model file (.joblib)
        preprocessor_path : str
            Path to the preprocessor file (.joblib)
        """
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.threshold = OPTIMAL_THRESHOLD
        
    def set_threshold(self, threshold: float) -> None:
        """
        Set the classification threshold.
        
        Parameters
        ----------
        threshold : float
            Classification threshold (0 to 1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict default probabilities.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
            
        Returns
        -------
        np.ndarray
            Array of default probabilities
        """
        X_processed = self.preprocessor.transform(X)
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        return probabilities
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict default (binary classification).
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
            
        Returns
        -------
        np.ndarray
            Array of binary predictions (0=Non-Default, 1=Default)
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions
    
    def get_risk_segment(self, probability: float) -> str:
        """
        Determine risk segment based on default probability.
        
        Parameters
        ----------
        probability : float
            Default probability
            
        Returns
        -------
        str
            Risk segment name
        """
        for segment, (low, high) in RISK_SEGMENTS.items():
            if low <= probability < high:
                return segment
        return 'Very High'
    
    def assess_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive risk assessment.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
            
        Returns
        -------
        pd.DataFrame
            Risk assessment results including probability, prediction, and segment
        """
        probabilities = self.predict_proba(X)
        predictions = self.predict(X)
        
        results = pd.DataFrame({
            'default_probability': probabilities,
            'predicted_default': predictions,
            'risk_segment': [self.get_risk_segment(p) for p in probabilities]
        })
        
        return results
    
    def get_recommendation(self, probability: float) -> Dict[str, str]:
        """
        Get lending recommendation based on default probability.
        
        Parameters
        ----------
        probability : float
            Default probability
            
        Returns
        -------
        dict
            Recommendation including action and rationale
        """
        segment = self.get_risk_segment(probability)
        
        recommendations = {
            'Very Low': {
                'action': 'Auto-Approve',
                'rationale': 'Very low default risk. Standard terms applicable.',
                'interest_adjustment': 'Standard rate'
            },
            'Low': {
                'action': 'Approve',
                'rationale': 'Low default risk. Standard review sufficient.',
                'interest_adjustment': 'Standard rate'
            },
            'Medium': {
                'action': 'Enhanced Review',
                'rationale': 'Moderate default risk. Additional documentation required.',
                'interest_adjustment': '+1-2% premium'
            },
            'High': {
                'action': 'Conditional Approval',
                'rationale': 'High default risk. Collateral or guarantor required.',
                'interest_adjustment': '+3-4% premium'
            },
            'Very High': {
                'action': 'Decline',
                'rationale': 'Very high default risk. Application not recommended.',
                'interest_adjustment': 'N/A'
            }
        }
        
        return recommendations.get(segment, recommendations['Very High'])


def predict_from_dict(applicant_data: Dict) -> Dict:
    """
    Convenience function to predict risk from a dictionary of applicant data.
    
    Parameters
    ----------
    applicant_data : dict
        Dictionary containing applicant features
        
    Returns
    -------
    dict
        Risk assessment results
    """
    predictor = CreditRiskPredictor()
    
    # Convert to DataFrame
    df = pd.DataFrame([applicant_data])
    
    # Get prediction
    proba = predictor.predict_proba(df)[0]
    prediction = predictor.predict(df)[0]
    segment = predictor.get_risk_segment(proba)
    recommendation = predictor.get_recommendation(proba)
    
    return {
        'default_probability': round(proba, 4),
        'predicted_default': bool(prediction),
        'risk_segment': segment,
        'recommendation': recommendation
    }


# Example usage
if __name__ == "__main__":
    # Example applicant data
    sample_applicant = {
        'checking_status': 'A11',  # < 0 DM
        'duration': 24,
        'credit_history': 'A32',  # Existing credits paid till now
        'purpose': 'A43',  # Radio/TV
        'credit_amount': 5000,
        'savings_status': 'A61',  # < 100 DM
        'employment': 'A73',  # 1-4 years
        'installment_commitment': 3,
        'personal_status': 'A93',  # Male single
        'other_parties': 'A101',  # None
        'residence_since': 3,
        'property_magnitude': 'A121',  # Real estate
        'age': 35,
        'other_payment_plans': 'A143',  # None
        'housing': 'A152',  # Own
        'existing_credits': 1,
        'job': 'A173',  # Skilled employee
        'num_dependents': 1,
        'own_telephone': 'A192',  # Yes
        'foreign_worker': 'A201'  # Yes
    }
    
    try:
        result = predict_from_dict(sample_applicant)
        print("\n=== Credit Risk Assessment ===")
        print(f"Default Probability: {result['default_probability']:.2%}")
        print(f"Predicted Default: {'Yes' if result['predicted_default'] else 'No'}")
        print(f"Risk Segment: {result['risk_segment']}")
        print(f"\nRecommendation:")
        print(f"  Action: {result['recommendation']['action']}")
        print(f"  Rationale: {result['recommendation']['rationale']}")
        print(f"  Interest Adjustment: {result['recommendation']['interest_adjustment']}")
    except FileNotFoundError:
        print("Error: Model files not found. Please run the notebook first to train the model.")
