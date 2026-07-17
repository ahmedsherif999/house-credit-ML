# src/preprocessing.py
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def get_preprocessing_pipeline(x_train, missing_threshold=50):
    print("⚙️ Building Preprocessing Pipeline...")
    
    missing_pct = x_train.isnull().sum() / len(x_train) * 100
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
    
    working_x = x_train.drop(columns=cols_to_drop, errors='ignore')
    
    num_cols = working_x.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = working_x.select_dtypes(include=[np.object_]).columns.tolist()
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    Ct = ColumnTransformer([
        ('nums', num_pipeline, num_cols),
        ('cats', cat_pipeline, cat_cols)
    ])
    
    return Ct, num_cols, cols_to_drop