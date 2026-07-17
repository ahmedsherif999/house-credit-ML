# src/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def load_raw_and_merge(data_dir="../data/"):
    print("📥 Loading raw datasets...")
    train = pd.read_csv(f"{data_dir}application_train.csv")
    test = pd.read_csv(f"{data_dir}application_test.csv")
    bureau_balance = pd.read_csv(f"{data_dir}bureau_balance.csv")
    bureau = pd.read_csv(f"{data_dir}bureau.csv")
    credit_balance = pd.read_csv(f"{data_dir}credit_card_balance.csv")
    installments_payments = pd.read_csv(f"{data_dir}installments_payments.csv")
    pos_cash = pd.read_csv(f"{data_dir}POS_CASH_balance.csv")
    
    # 1. Processing Bureau Balance
    print("🔄 Processing Bureau Balance...")
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
    
    num_cols = merged_bureau.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove('SK_ID_CURR')
    if 'SK_ID_BUREAU' in num_cols:
        num_cols.remove('SK_ID_BUREAU')
        
    bureau_agg_dict = {col: ["min", 'max', 'mean', 'std'] for col in num_cols}
    client_bureau_agg = merged_bureau.groupby('SK_ID_CURR').agg(bureau_agg_dict).reset_index()
    client_bureau_agg.columns = ['_'.join(col).strip() if col[0] != 'SK_ID_CURR' else col[0] for col in client_bureau_agg.columns.values]
    
    merged_bureau_train = pd.merge(train, client_bureau_agg, on='SK_ID_CURR', how='left')
    merged_bureau_test = pd.merge(test, client_bureau_agg, on='SK_ID_CURR', how='left')
    
    # 2. Processing Credit Card Balance
    print("🔄 Processing Credit Card Balance...")
    num_cols_c = credit_balance.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_c.remove('SK_ID_PREV')
    if 'SK_ID_CURR' in num_cols_c:
        num_cols_c.remove('SK_ID_CURR')
        
    credit_balance_agg_dict = {col: ["min", 'max', 'std', 'mean'] for col in num_cols_c}
    credit_balance_merge = credit_balance.groupby('SK_ID_CURR').agg(credit_balance_agg_dict).reset_index()
    credit_balance_merge.columns = ['_'.join(col).strip() if col[0] != 'SK_ID_CURR' else col[0] for col in credit_balance_merge.columns.values]
    
    merged_bur_credit_train = pd.merge(merged_bureau_train, credit_balance_merge, on='SK_ID_CURR', how='left')
    merged_bur_credit_test = pd.merge(merged_bureau_test, credit_balance_merge, on='SK_ID_CURR', how='left')
    
    # 3. Processing Installments Payments
    print("🔄 Processing Installments Payments...")
    num_cols_ins = installments_payments.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_ins.remove('SK_ID_PREV')
    if 'SK_ID_CURR' in num_cols_ins:
        num_cols_ins.remove('SK_ID_CURR')
        
    installments_payments_agg_dict = {col: ["min", 'max', 'std', 'mean'] for col in num_cols_ins}
    installments_payments_merge = installments_payments.groupby('SK_ID_CURR').agg(installments_payments_agg_dict).reset_index()
    installments_payments_merge.columns = ['_'.join(col).strip() if col[0] != 'SK_ID_CURR' else col[0] for col in installments_payments_merge.columns.values]
    
    merged_bur_credit_ins_train = pd.merge(merged_bur_credit_train, installments_payments_merge, on='SK_ID_CURR', how='left')
    merged_bur_credit_ins_test = pd.merge(merged_bur_credit_test, installments_payments_merge, on='SK_ID_CURR', how='left')
    
    # 4. Processing POS CASH Balance
    print("🔄 Processing POS CASH Balance...")
    num_cols_pos = pos_cash.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_pos.remove('SK_ID_PREV')
    if 'SK_ID_CURR' in num_cols_pos:
        num_cols_pos.remove('SK_ID_CURR')
        
    pos_cash_agg_dict = {col: ["min", 'max', 'std', 'mean'] for col in num_cols_pos}
    pos_cash_merge = pos_cash.groupby('SK_ID_CURR').agg(pos_cash_agg_dict).reset_index()
    pos_cash_merge.columns = ['_'.join(col).strip() if col[0] != 'SK_ID_CURR' else col[0] for col in pos_cash_merge.columns.values]
    
    master_train = pd.merge(merged_bur_credit_ins_train, pos_cash_merge, on='SK_ID_CURR', how='left')
    master_test = pd.merge(merged_bur_credit_ins_test, pos_cash_merge, on='SK_ID_CURR', how='left')
    
    return master_train, master_test