import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate():
    print("Loading engineered Parquet data...")
    master_train = pd.read_parquet('data/master_train.parquet')
    master_test = pd.read_parquet('data/master_test.parquet')

    print("Splitting Data...")
    x = master_train.drop('TARGET', axis=1)
    y = master_train["TARGET"]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # حفظ الـ ID لو احتجناه للـ submission بعدين
    test_ids = master_test["SK_ID_CURR"]

    x_train = x_train.drop("SK_ID_CURR", axis=1)
    x_val = x_val.drop("SK_ID_CURR", axis=1)
    test = master_test.drop("SK_ID_CURR", axis=1)

    print("Dropping columns with >50% missing values...")
    missing_pct = x_train.isnull().sum() / len(x_train) * 100
    cols_to_drop = missing_pct[missing_pct > 50].index
    x_train = x_train.drop(columns=cols_to_drop)
    x_val = x_val.drop(columns=cols_to_drop)
    test = test.drop(columns=cols_to_drop) # مهم نشيلهم من الـ test كمان

    print("Clipping outliers...")
    num_cols = x_train.select_dtypes(include=[np.number]).columns
    cat_cols = x_train.select_dtypes(include=[np.object_]).columns

    for col in num_cols:
        inlier_cap = x_train[col].quantile(0.99)
        x_train[col] = x_train[col].clip(upper=inlier_cap)
        x_val[col] = x_val[col].clip(upper=inlier_cap)
        test[col] = test[col].clip(upper=inlier_cap)

    print("Building Preprocessing Pipeline...")
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ('nums', num_pipeline, num_cols),
        ('cats', cat_pipeline, cat_cols)
    ])

    print("Transforming Data...")
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_val_transformed = preprocessor.transform(x_val)
    test_transformed = preprocessor.transform(test)

    # Class weight calculation
    weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print("Training XGBoost Model...")
    xg_model = XGBClassifier(
        n_estimators=500, 
        max_depth=6, 
        learning_rate=0.03, 
        scale_pos_weight=weight,
        n_jobs=-1,
        random_state=42
    )
    
    xg_model.fit(x_train_transformed, y_train)

    print("Evaluating Model...")
    preds = xg_model.predict(x_val_transformed)
    probabilities = xg_model.predict_proba(x_val_transformed)[:, 1]

    print("\n" + "="*30)
    print("=== XGBoost Classification Report ===")
    print("="*30)
    print(classification_report(y_val, preds))
    print(f"Accuracy:      {accuracy_score(y_val, preds):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_val, probabilities):.4f}")
    print("="*30)

if __name__ == "__main__":
    train_and_evaluate()