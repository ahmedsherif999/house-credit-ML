# src/train.py
import os
import joblib
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import lightgbm as lgb

from data_loader import load_raw_and_merge
from feature_engineering import add_domain_features
from preprocessing import get_preprocessing_pipeline

def clean_column_names(df):
    df.columns = [re.sub(r'[\\[\\]\\{\\}\\:\\,]', '_', col) for col in df.columns]
    return df

def main():
    master_train, master_test = load_raw_and_merge(data_dir="data/")
    
    master_train = add_domain_features(master_train)
    master_test = add_domain_features(master_test)
    
    
    test_ids = master_test["SK_ID_CURR"]
    
    x = master_train.drop('TARGET', axis=1)
    y = master_train["TARGET"]
    
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    
    x_train = x_train.drop("SK_ID_CURR", axis=1)
    x_val = x_val.drop("SK_ID_CURR", axis=1)
    test = master_test.drop("SK_ID_CURR", axis=1)
    
    Ct, num_cols, cols_to_drop = get_preprocessing_pipeline(x_train)
    
    x_train = x_train.drop(columns=cols_to_drop, errors='ignore')
    x_val = x_val.drop(columns=cols_to_drop, errors='ignore')
    test = test.drop(columns=cols_to_drop, errors='ignore')
    
    print("✂️ Clipping Outliers...")
    for i in num_cols:
        inliers = x_train[i].quantile(0.99)
        x_train[i] = x_train[i].clip(upper=inliers)
        x_val[i] = x_val[i].clip(upper=inliers)
        test[i] = test[i].clip(upper=inliers)
        
    for col in num_cols:
        x_train[col] = x_train[col].astype('float32')
        x_val[col] = x_val[col].astype('float32')
        test[col] = test[col].astype('float32')
        
    print("🪄 Transforming matrices...")
    x_train_transformed = Ct.fit_transform(x_train)
    x_val_transformed = Ct.transform(x_val)
    test_transformed = Ct.transform(test)
    
    os.makedirs("../models", exist_ok=True)
    joblib.dump(Ct, "../models/preprocessor.joblib")
    
    print("🚀 Training LightGBM Model...")
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, num_leaves=31, max_depth=8,
        min_data_in_leaf=20, class_weight='balanced', random_state=42, n_jobs=-1
    )
    lgbm_model.fit(
        x_train_transformed, y_train, eval_set=[(x_val_transformed, y_val)],
        eval_metric='auc', callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )
    
    joblib.dump(lgbm_model, "../models/lgbm_model.joblib")
    
    lgbm_probas = lgbm_model.predict_proba(x_val_transformed)[:, 1]
    print("\n" + "="*30)
    print(f"✅ Final LightGBM ROC-AUC: {roc_auc_score(y_val, lgbm_probas):.4f}")
    print("="*30)
    
    print("💾 Saving Kaggle Submissions...")
    test_probabilities = lgbm_model.predict_proba(test_transformed)[:, 1]
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_probabilities})
    submission.to_csv('../data/submission_lgbm.csv', index=False)
    print("🎉 Submission file successfully saved!")

if __name__ == '__main__':
    main()