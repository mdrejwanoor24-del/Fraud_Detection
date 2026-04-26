import joblib
import numpy as np
import pandas as pd

# Load model
data = joblib.load("best_model.pkl")
model = data["model"]
threshold = data["threshold"]
features = data["features"]

def predict_fraud(features_dict):
    """Predict fraud for a transaction.
    
    Args:
        features_dict: dict with feature values matching training data
    
    Returns:
        str: Prediction result with probability
    """
    # Create feature array in correct order
    feature_array = [features_dict.get(f, 0) for f in features]
    X = np.array(feature_array).reshape(1, -1)
    
    prob = model.predict_proba(X)[:, 1][0]
    
    if prob >= threshold:
        return f"Fraud (Prob: {prob:.2f})"
    else:
        return f"Safe (Prob: {prob:.2f})"


# Example usage
if __name__ == "__main__":
    # Sample transaction (matching feature order from training)
    sample = {
        'step': 10,
        'type': 4,  # TRANSFER encoded value
        'amount': 50000.0,
        'oldbalanceOrg': 100000.0,
        'newbalanceOrig': 50000.0,
        'oldbalanceDest': 0.0,
        'newbalanceDest': 50000.0,
        'balance_diff_orig': 50000.0,
        'balance_diff_dest': 50000.0,
        'hour': 10
    }
    
    result = predict_fraud(sample)
    print(f"Prediction: {result}")
    