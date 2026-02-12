import pandas as pd
import numpy as np
import joblib
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

import category_encoders as ce
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# =========================
# ARTIFACT FOLDER
# =========================

os.makedirs("artifacts", exist_ok=True)

# =========================
# CONFIG
# =========================

TARGET = "sale_price"
SEED = 42

ORDINAL_COLS = ["submarket"]

TARGET_ENC_COLS = [
    "city","zoning","subdivision","join_status"
]

DROP_BASE = ["id","sale_warning","latitude","longitude"]

DROP_AFTER = [
    "join_year","sale_date","sqft_lot","garb_sqft","gara_sqft",
    "bath_full","bath_3qtr","bath_half","beds","present_use",
    "sale_day","sqft_1","imp_val","year_reno","fbsmt_grade",
    "view_otherwater","sale_nbr","view_sound","grade"
]

# =========================
# FEATURE ENGINEERING
# =========================

class HouseFE(BaseEstimator, TransformerMixin):

    def fit(self,X,y=None):
        return self

    def transform(self,df):

        d = df.copy()

        d = d.drop(columns=DROP_BASE, errors="ignore")

        d["sale_date"] = pd.to_datetime(d["sale_date"])

        d["sale_year"] = d["sale_date"].dt.year
        d["sale_month"] = d["sale_date"].dt.month
        d["sale_day"] = d["sale_date"].dt.day

        d["age_at_sale"] = d["sale_year"] - d["year_built"]
        d["reno_age_at_sale"] = d["sale_year"] - d["year_reno"]
        d["is_renovated"] = (d["year_reno"] > d["year_built"]).astype(int)

        d["lot_sqft_ratio"] = d["sqft"]/(d["sqft_lot"]+1)
        d["garage_total"] = d["garb_sqft"] + d["gara_sqft"]

        d["bath_total"] = (
            d["bath_full"] + d["bath_3qtr"] + 0.5*d["bath_half"]
        )

        d["sqft_per_bed"] = d["sqft"]/(d["beds"]+1)
        d["sqft_per_bath"] = d["sqft"]/(d["bath_total"]+1)

        d = d.drop(columns=DROP_AFTER, errors="ignore")

        return d

# =========================
# METRICS
# =========================

def pinball(y,q,a):
    d=y-q
    return np.mean(np.maximum(a*d,(a-1)*d))

def interval_metrics(y,l,u):
    cov=np.mean((y>=l)&(y<=u))
    width=np.mean(u-l)
    mae_mid=mean_absolute_error(y,(l+u)/2)
    return cov,width,mae_mid

# =========================
# LOAD DATA
# =========================

train_df = pd.read_csv("data/dataset.csv")
test_df  = pd.read_csv("data/test.csv")

train_ids = train_df["id"]   # ✅ ADDED
test_ids = test_df["id"]

X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]

# =========================
# VALIDATION SPLIT
# =========================

Xtr,Xval,ytr,yval = train_test_split(
    X,y,test_size=0.2,random_state=SEED
)

# =========================
# PREPROCESS FIT ON TRAIN SPLIT
# =========================

fe = HouseFE()

Xtr = fe.fit_transform(Xtr)
Xval = fe.transform(Xval)

# =========================
# ENCODING
# =========================

target_enc = ce.TargetEncoder(
    cols=TARGET_ENC_COLS,
    smoothing=20,
    min_samples_leaf=50
)

Xtr = target_enc.fit_transform(Xtr,ytr)
Xval = target_enc.transform(Xval)

ord_enc = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

Xtr[ORDINAL_COLS] = ord_enc.fit_transform(Xtr[ORDINAL_COLS])
Xval[ORDINAL_COLS] = ord_enc.transform(Xval[ORDINAL_COLS])

imp = SimpleImputer(strategy="median")

Xtr = pd.DataFrame(imp.fit_transform(Xtr),columns=Xtr.columns)
Xval = pd.DataFrame(imp.transform(Xval),columns=Xval.columns)

# =========================
# VALIDATION EVALUATION + SAVE METRICS
# =========================

metrics_rows = []

for name, builder in {
    "LightGBM": build_lgbm,
    "XGBoost": build_xgb,
    "CatBoost": build_catboost
}.items():

    low = builder(0.1)
    up  = builder(0.9)

    low.fit(Xtr,ytr)
    up.fit(Xtr,ytr)

    l = low.predict(Xval)
    u = up.predict(Xval)

    l,u = np.minimum(l,u),np.maximum(l,u)

    cov,width,mae_mid = interval_metrics(yval,l,u)
    p10 = pinball(yval,l,0.1)
    p90 = pinball(yval,u,0.9)
    
    print(f"\n==== {name} VALIDATION ====")
    print("coverage:",cov)
    print("width:",width)
    print("mae_mid:",mae_mid)
    print("pinball10:",p10)
    print("pinball90:",p90)

    metrics_rows.append({
        "model": name,
        "coverage": cov,
        "avg_width": width,
        "mae_mid": mae_mid,
        "pinball10": p10,
        "pinball90": p90
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv("artifacts/metrics_validation.csv", index=False)

# =========================
# FULL TRAIN PREPROCESS
# =========================

X_full = fe.fit_transform(X)

X_full_save = X_full.copy()
X_full_save.insert(0, "id", train_ids.values)   # ✅ ADDED
X_full_save[TARGET] = y.values
X_full_save.to_csv("artifacts/dataset_Final.csv", index=False)

print("✅ dataset_Final.csv saved")

X_full = target_enc.fit_transform(X_full,y)
X_full[ORDINAL_COLS] = ord_enc.fit_transform(X_full[ORDINAL_COLS])
X_full = pd.DataFrame(imp.fit_transform(X_full),columns=X_full.columns)

# =========================
# TEST PREPROCESS
# =========================

X_test = fe.transform(test_df)

X_test_save = X_test.copy()
X_test_save.insert(0, "id", test_ids.values)   # ✅ ADDED
X_test_save.to_csv("artifacts/test_Final.csv", index=False)

print("✅ test_Final.csv saved")

X_test = target_enc.transform(X_test)
X_test[ORDINAL_COLS] = ord_enc.transform(X_test[ORDINAL_COLS])
X_test = pd.DataFrame(imp.transform(X_test),columns=X_test.columns)

# =========================
# FINAL TRAIN + TEST PREDICT
# =========================

for name, builder in {
    "LightGBM": build_lgbm,
    "XGBoost": build_xgb,
    "CatBoost": build_catboost
}.items():

    low = builder(0.1)
    up  = builder(0.9)

    low.fit(X_full,y)
    up.fit(X_full,y)

    l = low.predict(X_test)
    u = up.predict(X_test)

    l,u = np.minimum(l,u),np.maximum(l,u)

    pd.DataFrame({
        "id": test_ids,
        "pi_lower": l,
        "pi_upper": u
    }).to_csv(f"artifacts/submission_{name}.csv",index=False)

    print(f"submission_{name}.csv created")
