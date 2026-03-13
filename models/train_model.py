import os
import joblib
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
import pandas as pd
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("Loading sampled datasets...")

files = glob.glob("../data/sampled/sample_*.parquet")
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

print("Sampled datasets loaded and concatenated.>>>")
print("Dataset shape:", df.shape)

# ---- FEATURES ----
features = [
    "trip_miles",
    "trip_time",
    "PULocationID",
    "DOLocationID",
    "tips",
    "tolls",
    "congestion_surcharge",
    "airport_fee",
    "driver_pay",
    "trip_minutes",
    "trip_speed",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend"
]

target = "base_passenger_fare"

X = df[features]
y = df[target]
print("Features and target variable separated.>>>")

# ---- TRAIN/TEST SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"Train size: {X_train.shape}")
print(f"Test size:  {X_test.shape}")

# ---- TRAIN MODEL ----
print("Training model...")

model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=100,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='mae',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

print("\n✅ Model training complete!")
print(f"Best iteration: {model.best_iteration_}")

# ---- EVALUATE ----
preds = model.predict(X_test)

mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)

print("=" * 40)
print("     MODEL EVALUATION RESULTS")
print("=" * 40)
print(f"  MAE  : ${mae:.2f}")
print(f"  RMSE : ${rmse:.2f}")
print(f"  R²   : {r2:.4f}")
print("=" * 40)

# ---- SAVE MODEL ----
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/fare_model.pkl")
print("\n✅ Model saved to ../models/fare_model.pkl")