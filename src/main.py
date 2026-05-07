import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
import os

warnings.filterwarnings('ignore')

def add_domain_features(df):
    print("Adding Domain Features...")

    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['GOODS_PRICE_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['EMPLOYED_AGE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    return df

def build_features():
    print("🚀 Step 1: Loading raw CSV files...")
    train = pd.read_csv("data/application_train.csv")
    test = pd.read_csv("data/application_test.csv")
    bureau_balance = pd.read_csv("data/bureau_balance.csv")
    bureau = pd.read_csv("data/bureau.csv")
    credit_balance = pd.read_csv("data/credit_card_balance.csv")
    installments_payments = pd.read_csv("data/installments_payments.csv")
    pos_cash = pd.read_csv("data/POS_CASH_balance.csv")

    train = add_domain_features(train)
    test = add_domain_features(test)

    print("Step 2: Processing Bureau & Bureau Balance...")
    ord_rank = ["C", "X", '0', '1', '2', '3', '4', '5']
    encoder = OrdinalEncoder(categories=[ord_rank])
    bureau_balance["encoded_status"] = encoder.fit_transform(bureau_balance[['STATUS']])
    
    bb_aggregations = bureau_balance.groupby("SK_ID_BUREAU").agg({
        'MONTHS_BALANCE': ["min", 'max', 'size'],
        'encoded_status': ["mean", 'max', 'std']
    })
    bb_aggregations.columns = ['_'.join(col).strip() for col in bb_aggregations.columns.values]
    bb_aggregations = bb_aggregations.reset_index()
    
    merged_bureau = pd.merge(bureau, bb_aggregations, on='SK_ID_BUREAU', how='left')
    
    num_cols_bur = merged_bureau.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_bur.remove('SK_ID_CURR')
    if 'SK_ID_BUREAU' in num_cols_bur: num_cols_bur.remove('SK_ID_BUREAU')
        
    bureau_agg_dict = {col: ["min", 'max', 'mean', 'std'] for col in num_cols_bur}
    client_bureau_agg = merged_bureau.groupby('SK_ID_CURR').agg(bureau_agg_dict).reset_index()
    client_bureau_agg.columns = ['_'.join(col).strip() if col[0] != 'SK_ID_CURR' else col[0] for col in client_bureau_agg.columns.values]
    
    train = pd.merge(train, client_bureau_agg, on='SK_ID_CURR', how='left')
    test = pd.merge(test, client_bureau_agg, on='SK_ID_CURR', how='left')

    print("Step 3: Processing Credit Card, Installments & POS...")
    def aggregate_and_merge(df, base_train, base_test, prefix):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SK_ID_PREV' in num_cols: num_cols.remove('SK_ID_PREV')
        if 'SK_ID_CURR' in num_cols: num_cols.remove('SK_ID_CURR')
        
        agg_dict = {col: ["min", 'max', 'std', 'mean'] for col in num_cols}
        agg_df = df.groupby('SK_ID_CURR').agg(agg_dict).reset_index()
        agg_df.columns = ['_'.join(col).strip() if col[0] != 'SK_ID_CURR' else col[0] for col in agg_df.columns.values]
        
        return pd.merge(base_train, agg_df, on='SK_ID_CURR', how='left'), pd.merge(base_test, agg_df, on='SK_ID_CURR', how='left')

    train, test = aggregate_and_merge(credit_balance, train, test, "credit")
    train, test = aggregate_and_merge(installments_payments, train, test, "ins")
    train, test = aggregate_and_merge(pos_cash, train, test, "pos")

    return train, test

def train_model(train_df, test_df):
    print("🔥 Step 4: Training LightGBM Model (The Beast)...")
    
    y = train_df['TARGET']
    test_ids = test_df['SK_ID_CURR']
    
    X = train_df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    X_test = test_df.drop(['SK_ID_CURR'], axis=1)
    
    for col in X.select_dtypes('object').columns:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    lgbm = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=34,
        max_depth=10,
        feature_fraction=0.8,
        bagging_fraction=0.7,
        bagging_freq=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
    )

    print(f"\n Validation AUC: {roc_auc_score(y_val, lgbm.predict_proba(X_val)[:, 1]):.4f}")

    # السابميشن
    print("Step 5: Generating Submission...")
    test_probs = lgbm.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_probs})
    submission.to_csv('data/submission_final.csv', index=False)
    print(" All done! File saved: data/submission_final.csv")

if __name__ == "__main__":
    master_train, master_test = build_features()
    train_model(master_train, master_test)