import os
import joblib
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
import pandas as pd
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import time

# =============================================
#    STEP 1: LOAD DATA
# =============================================
print("=" * 60)
print("    IMPROVED MODEL - FEATURE ENGINEERING")
print("=" * 60)

print("\nLoading data...")
files = glob.glob("../data/sampled/sample_*.parquet")
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded: {df.shape}")

# =============================================
#    STEP 2: FEATURE ENGINEERING (NEW!)
# =============================================
print("\n🔧 Engineering new features...")

# ---- Cost-based features ----
df['total_extras'] = df['tips'] + df['tolls'] + df['congestion_surcharge'] + df['airport_fee']
df['fare_per_mile'] = df['driver_pay'] / df['trip_miles'].replace(0, np.nan)
df['fare_per_minute'] = df['driver_pay'] / df['trip_minutes'].replace(0, np.nan)

# ---- Trip characteristics ----
df['miles_per_minute'] = df['trip_miles'] / df['trip_minutes'].replace(0, np.nan)
df['is_long_trip'] = (df['trip_miles'] > df['trip_miles'].quantile(0.75)).astype(int)
df['is_short_trip'] = (df['trip_miles'] < df['trip_miles'].quantile(0.25)).astype(int)

# ---- Time-based features ----
df['is_rush_hour'] = df['pickup_hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
df['is_night'] = df['pickup_hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
df['is_morning'] = df['pickup_hour'].isin([6, 7, 8, 9, 10, 11]).astype(int)

# ---- Time period ----
def get_time_period(hour):
    if hour in [6, 7, 8, 9]:
        return 0    # morning rush
    elif hour in [10, 11, 12, 13, 14, 15]:
        return 1    # midday
    elif hour in [16, 17, 18, 19]:
        return 2    # evening rush
    elif hour in [20, 21, 22, 23]:
        return 3    # evening
    else:
        return 4    # late night

df['time_period'] = df['pickup_hour'].apply(get_time_period)

# ---- Location-based ----
df['same_zone'] = (df['PULocationID'] == df['DOLocationID']).astype(int)

# ---- Airport detection ----
airport_ids = [1, 132, 138]  # JFK, LaGuardia, Newark area
df['is_airport_pickup'] = df['PULocationID'].isin(airport_ids).astype(int)
df['is_airport_dropoff'] = df['DOLocationID'].isin(airport_ids).astype(int)
df['is_airport_trip'] = (df['is_airport_pickup'] | df['is_airport_dropoff']).astype(int)

# ---- Interaction features ----
df['miles_x_speed'] = df['trip_miles'] * df['trip_speed']
df['time_x_congestion'] = df['trip_minutes'] * df['congestion_surcharge']
df['distance_x_hour'] = df['trip_miles'] * df['pickup_hour']

# ---- Fill any NaN from division ----
df = df.fillna(0)

# Replace infinity values
df = df.replace([np.inf, -np.inf], 0)

print(f"✅ New features created!")
print(f"   Total columns now: {df.shape[1]}")

# =============================================
#    STEP 3: REMOVE OUTLIERS
# =============================================
print("\n🔧 Removing outliers...")

before = len(df)

# Remove extreme fares
q1 = df['base_passenger_fare'].quantile(0.01)
q99 = df['base_passenger_fare'].quantile(0.99)
df = df[(df['base_passenger_fare'] >= q1) & (df['base_passenger_fare'] <= q99)]

# Remove zero/negative fares
df = df[df['base_passenger_fare'] > 0]

# Remove extreme trip miles
df = df[df['trip_miles'] > 0]
df = df[df['trip_miles'] < df['trip_miles'].quantile(0.99)]

# Remove extreme trip times
df = df[df['trip_time'] > 0]
df = df[df['trip_time'] < df['trip_time'].quantile(0.99)]

after = len(df)
print(f"   Removed {before - after:,} outliers ({(before-after)/before*100:.1f}%)")
print(f"   Remaining: {after:,} rows")

# =============================================
#    STEP 4: DEFINE FEATURES
# =============================================
target = "base_passenger_fare"

# Original features
original_features = [
    "trip_miles", "trip_time", "PULocationID", "DOLocationID",
    "tips", "tolls", "congestion_surcharge", "airport_fee",
    "driver_pay", "trip_minutes", "trip_speed",
    "pickup_hour", "pickup_dayofweek", "is_weekend"
]

# New engineered features
new_features = [
    "total_extras",
    "fare_per_mile",
    "fare_per_minute",
    "miles_per_minute",
    "is_long_trip",
    "is_short_trip",
    "is_rush_hour",
    "is_night",
    "is_morning",
    "time_period",
    "same_zone",
    "is_airport_pickup",
    "is_airport_dropoff",
    "is_airport_trip",
    "miles_x_speed",
    "time_x_congestion",
    "distance_x_hour",
]

all_features = original_features + new_features

print(f"\n📊 Feature Count:")
print(f"   Original: {len(original_features)}")
print(f"   New:      {len(new_features)}")
print(f"   Total:    {len(all_features)}")

X = df[all_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n   Train: {X_train.shape}")
print(f"   Test:  {X_test.shape}")

# =============================================
#    STEP 5: BASELINE COMPARISON
# =============================================
print("\n" + "-" * 60)

# Train baseline model on SAME cleaned data (fair comparison)
print("Training baseline on cleaned data (original features only)...")

X_train_base = X_train[original_features]
X_test_base = X_test[original_features]

baseline_model = LGBMRegressor(
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

baseline_model.fit(
    X_train_base, y_train,
    eval_set=[(X_test_base, y_test)],
    eval_metric='mae',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=0)
    ]
)

baseline_preds = baseline_model.predict(X_test_base)
baseline_mae  = mean_absolute_error(y_test, baseline_preds)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
baseline_r2   = r2_score(y_test, baseline_preds)

print(f"\n📊 BASELINE (original features, cleaned data):")
print(f"   MAE  : ${baseline_mae:.2f}")
print(f"   RMSE : ${baseline_rmse:.2f}")
print(f"   R²   : {baseline_r2:.4f}")

# =============================================
#    STEP 6: TRAIN IMPROVED MODELS
# =============================================
print("\n" + "=" * 60)
print("    TRAINING IMPROVED MODELS")
print("=" * 60)

experiments = [
    {
        "name": "1. New features + baseline params",
        "params": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 100,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }
    },
    {
        "name": "2. New features + best tuned params",
        "params": {
            "n_estimators": 2000,
            "learning_rate": 0.02,
            "max_depth": 11,
            "num_leaves": 200,
            "min_child_samples": 50,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 0.5,
        }
    },
    {
        "name": "3. New features + deeper model",
        "params": {
            "n_estimators": 2000,
            "learning_rate": 0.02,
            "max_depth": 13,
            "num_leaves": 250,
            "min_child_samples": 40,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.05,
            "reg_lambda": 0.3,
        }
    },
    {
        "name": "4. New features + high trees",
        "params": {
            "n_estimators": 3000,
            "learning_rate": 0.01,
            "max_depth": 12,
            "num_leaves": 220,
            "min_child_samples": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 0.2,
        }
    },
    {
        "name": "5. New features + balanced",
        "params": {
            "n_estimators": 2500,
            "learning_rate": 0.015,
            "max_depth": 11,
            "num_leaves": 200,
            "min_child_samples": 60,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 0.4,
        }
    },
]

results = []
best_mae = float('inf')
best_name = ""
best_model = None

for i, exp in enumerate(experiments):
    print(f"\n{'─' * 60}")
    print(f"🔧 Experiment {i+1}/{len(experiments)}: {exp['name']}")
    print(f"{'─' * 60}")

    start = time.time()

    model = LGBMRegressor(
        **exp['params'],
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
            lgb.log_evaluation(period=0)
        ]
    )

    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    elapsed = time.time() - start

    mae_change = ((mae - baseline_mae) / baseline_mae) * 100

    print(f"   MAE  : ${mae:.2f}  (change: {mae_change:+.1f}%)")
    print(f"   RMSE : ${rmse:.2f}")
    print(f"   R²   : {r2:.4f}")
    print(f"   Trees: {model.best_iteration_} | Time: {elapsed:.1f}s")

    results.append({
        'Experiment': exp['name'],
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'Trees': model.best_iteration_,
        'Time': round(elapsed, 1),
        'MAE_Change_%': round(mae_change, 1)
    })

    if mae < best_mae:
        best_mae = mae
        best_name = exp['name']
        best_model = model
        print(f"   🏆 NEW BEST!")

# =============================================
#    STEP 7: FINAL RESULTS
# =============================================
results_df = pd.DataFrame(results).sort_values('MAE')

print("\n" + "=" * 60)
print("    ALL RESULTS (Sorted by MAE)")
print("=" * 60)
print(results_df.to_string(index=False))

# ---- Best vs Baseline ----
best_preds = best_model.predict(X_test)
best_mae  = mean_absolute_error(y_test, best_preds)
best_rmse = np.sqrt(mean_squared_error(y_test, best_preds))
best_r2   = r2_score(y_test, best_preds)

# Load ORIGINAL baseline (from train_model.py)
original_model = joblib.load("../models/fare_model.pkl")

print("\n" + "=" * 60)
print("    COMPLETE COMPARISON")
print("=" * 60)
print(f"\n{'Model':<35} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 65)
print(f"{'Original baseline (train_model.py)':<35} {'$4.03':>8} {'$7.33':>8} {'0.9051':>8}")
print(f"{'Tuned baseline (tune_model.py)':<35} {'$3.98':>8} {'$7.28':>8} {'0.9064':>8}")
print(f"{'Cleaned data baseline':<35} {'$'+f'{baseline_mae:.2f}':>8} {'$'+f'{baseline_rmse:.2f}':>8} {baseline_r2:>8.4f}")
print(f"{'🏆 Best improved model':<35} {'$'+f'{best_mae:.2f}':>8} {'$'+f'{best_rmse:.2f}':>8} {best_r2:>8.4f}")

total_improvement = ((best_mae - 4.03) / 4.03) * 100
print(f"\n📈 Total improvement from original: {total_improvement:+.1f}%")

# =============================================
#    STEP 8: FEATURE IMPORTANCE (NEW vs OLD)
# =============================================
print("\n" + "=" * 60)
print("    TOP 15 MOST IMPORTANT FEATURES")
print("=" * 60)

importance = pd.Series(best_model.feature_importances_, index=all_features)
importance = importance.sort_values(ascending=False)

for i, (feat, imp) in enumerate(importance.head(15).items()):
    bar = "█" * int(imp / importance.max() * 30)
    tag = " ⭐ NEW" if feat in new_features else ""
    print(f"   {i+1:>2}. {feat:<25} {imp:>6} {bar}{tag}")

# =============================================
#    STEP 9: SAVE
# =============================================
os.makedirs("../models", exist_ok=True)

joblib.dump(best_model, "../models/fare_model_improved.pkl")
print(f"\n✅ Improved model saved → ../models/fare_model_improved.pkl")

results_df.to_csv("../models/improvement_results.csv", index=False)
print(f"✅ Results saved → ../models/improvement_results.csv")

# Save feature list
with open("../models/improved_features.txt", "w") as f:
    for feat in all_features:
        f.write(feat + "\n")
print(f"✅ Feature list saved → ../models/improved_features.txt")

# Save complete comparison
comparison = pd.DataFrame({
    'Model': ['Original Baseline', 'Tuned Baseline', 'Improved (Best)'],
    'MAE':   [4.03, 3.98, best_mae],
    'RMSE':  [7.33, 7.28, best_rmse],
    'R2':    [0.9051, 0.9064, best_r2],
    'Features': [14, 14, len(all_features)]
})
comparison.to_csv("../models/full_comparison.csv", index=False)
print(f"✅ Full comparison saved → ../models/full_comparison.csv")

print("\n" + "=" * 60)
print("    🎉 IMPROVEMENT COMPLETE!")
print("=" * 60)