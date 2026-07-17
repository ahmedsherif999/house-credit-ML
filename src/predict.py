# src/predict.py
import joblib
import pandas as pd
from feature_engineering import add_domain_features

def predict_single_client(client_raw_dict, models_dir="../models/"):
    preprocessor = joblib.load(f"{models_dir}preprocessor.joblib")
    model = joblib.load(f"{models_dir}lgbm_model.joblib")
    
    client_df = pd.DataFrame([client_raw_dict])
    client_df = add_domain_features(client_df)
    
    client_transformed = preprocessor.transform(client_df)
    
    prob = model.predict_proba(client_transformed)[:, 1][0]
    return float(prob)