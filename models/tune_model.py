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
#          STEP 1: LOAD DATA
# =============================================
print("=" * 60)
print("       MANUAL HYPERPARAMETER TUNING")
print("=" * 60)

print("\nLoading sampled datasets...")

files = glob.glob("../data/sampled/sample_*.parquet")
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

print(f"✅ Dataset loaded: {df.shape}")

features = [
    "trip_miles", "trip_time", "PULocationID", "DOLocationID",
    "tips", "tolls", "congestion_surcharge", "airport_fee",
    "driver_pay", "trip_minutes", "trip_speed",
    "pickup_hour", "pickup_dayofweek", "is_weekend"
]
target = "base_passenger_fare"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Train: {X_train.shape}")
print(f"   Test:  {X_test.shape}")

# =============================================
#    STEP 2: BASELINE RESULTS (for comparison)
# =============================================
print("\n" + "-" * 60)
print("Loading baseline model for comparison...")

baseline_model = joblib.load("../models/fare_model.pkl")
baseline_preds = baseline_model.predict(X_test)

baseline_mae  = mean_absolute_error(y_test, baseline_preds)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
baseline_r2   = r2_score(y_test, baseline_preds)

print(f"\n📊 BASELINE RESULTS:")
print(f"   MAE  : ${baseline_mae:.2f}")
print(f"   RMSE : ${baseline_rmse:.2f}")
print(f"   R²   : {baseline_r2:.4f}")

# =============================================
#    STEP 3: DEFINE EXPERIMENTS
# =============================================
#
# TUNING STRATEGY:
#
# Exp 1: Baseline        → Same as original (for reference)
# Exp 2: More trees      → n_estimators 500 → 1000
# Exp 3: Lower LR        → learning_rate 0.05 → 0.03
# Exp 4: More trees + Lower LR combined
# Exp 5: Deeper trees    → max_depth 8 → 12, num_leaves 100 → 200
# Exp 6: Less overfitting → higher regularization
# Exp 7: More data per tree → subsample & colsample
# Exp 8: Best combo v1
# Exp 9: Best combo v2
# Exp 10: Aggressive     → many trees, very low LR
#

experiments = [
    # ---- Experiment 1: Baseline (reference) ----
    {
        "name": "1. Baseline (original)",
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

    # ---- Experiment 2: Just increase trees ----
    {
        "name": "2. More trees (1000)",
        "params": {
            "n_estimators": 1000,       # 500 → 1000
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

    # ---- Experiment 3: Just lower learning rate ----
    {
        "name": "3. Lower LR (0.03)",
        "params": {
            "n_estimators": 500,
            "learning_rate": 0.03,      # 0.05 → 0.03
            "max_depth": 8,
            "num_leaves": 100,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }
    },

    # ---- Experiment 4: More trees + Lower LR ----
    {
        "name": "4. More trees + Lower LR",
        "params": {
            "n_estimators": 1000,       # 500 → 1000
            "learning_rate": 0.03,      # 0.05 → 0.03
            "max_depth": 8,
            "num_leaves": 100,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }
    },

    # ---- Experiment 5: Deeper trees ----
    {
        "name": "5. Deeper trees",
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 12,            # 8 → 12
            "num_leaves": 200,          # 100 → 200
            "min_child_samples": 50,    # 100 → 50
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }
    },

    # ---- Experiment 6: Higher regularization ----
    {
        "name": "6. Higher regularization",
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 10,
            "num_leaves": 150,
            "min_child_samples": 80,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,           # 0.1 → 0.5
            "reg_lambda": 0.5,          # 0.1 → 0.5
        }
    },

    # ---- Experiment 7: More data per tree ----
    {
        "name": "7. More data sampling",
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 10,
            "num_leaves": 150,
            "min_child_samples": 80,
            "subsample": 0.9,           # 0.8 → 0.9
            "colsample_bytree": 0.9,    # 0.8 → 0.9
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }
    },

    # ---- Experiment 8: Best combo v1 ----
    {
        "name": "8. Best combo v1",
        "params": {
            "n_estimators": 1500,       # many trees
            "learning_rate": 0.02,      # slow learning
            "max_depth": 10,
            "num_leaves": 180,
            "min_child_samples": 60,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.05,
            "reg_lambda": 0.3,
        }
    },

    # ---- Experiment 9: Best combo v2 ----
    {
        "name": "9. Best combo v2",
        "params": {
            "n_estimators": 2000,       # even more trees
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

    # ---- Experiment 10: Aggressive ----
    {
        "name": "10. Aggressive (slow LR, many trees)",
        "params": {
            "n_estimators": 2000,
            "learning_rate": 0.01,      # very slow
            "max_depth": 12,
            "num_leaves": 220,
            "min_child_samples": 40,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 0.2,
        }
    },
]

# =============================================
#    STEP 4: RUN ALL EXPERIMENTS
# =============================================
print("\n" + "=" * 60)
print("       RUNNING EXPERIMENTS")
print("=" * 60)

results = []
best_mae = float('inf')
best_name = ""
best_model = None

total_start = time.time()

for i, exp in enumerate(experiments):
    print(f"\n{'─' * 60}")
    print(f"🔧 Experiment {i+1}/{len(experiments)}: {exp['name']}")
    print(f"{'─' * 60}")

    # Print key params being tested
    p = exp['params']
    print(f"   n_estimators={p['n_estimators']} | lr={p['learning_rate']} | "
          f"depth={p['max_depth']} | leaves={p['num_leaves']}")

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
            lgb.log_evaluation(period=0)    # silent training
        ]
    )

    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    elapsed = time.time() - start

    # Compare with baseline
    mae_change = ((mae - baseline_mae) / baseline_mae) * 100

    print(f"\n   📊 Results:")
    print(f"   MAE  : ${mae:.2f}  (baseline: ${baseline_mae:.2f} | change: {mae_change:+.1f}%)")
    print(f"   RMSE : ${rmse:.2f}")
    print(f"   R²   : {r2:.4f}")
    print(f"   Trees: {model.best_iteration_}")
    print(f"   Time : {elapsed:.1f}s")

    results.append({
        'Experiment': exp['name'],
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'Trees_Used': model.best_iteration_,
        'Time_Sec': round(elapsed, 1),
        'MAE_Change_%': round(mae_change, 1)
    })

    # Track best
    if mae < best_mae:
        best_mae = mae
        best_name = exp['name']
        best_model = model
        print(f"\n   🏆 NEW BEST MODEL!")

total_time = time.time() - total_start

# =============================================
#    STEP 5: RESULTS TABLE
# =============================================
print("\n" + "=" * 60)
print("       ALL RESULTS (Sorted by MAE)")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('MAE')

print(results_df.to_string(index=False))

# =============================================
#    STEP 6: BASELINE vs BEST COMPARISON
# =============================================
best_preds = best_model.predict(X_test)
best_mae_final  = mean_absolute_error(y_test, best_preds)
best_rmse_final = np.sqrt(mean_squared_error(y_test, best_preds))
best_r2_final   = r2_score(y_test, best_preds)

print("\n" + "=" * 60)
print("       BASELINE  vs  BEST TUNED")
print("=" * 60)

mae_imp  = ((best_mae_final - baseline_mae) / baseline_mae) * 100
rmse_imp = ((best_rmse_final - baseline_rmse) / baseline_rmse) * 100
r2_imp   = ((best_r2_final - baseline_r2) / baseline_r2) * 100

print(f"\n{'Metric':<10} {'Baseline':>12} {'Best Tuned':>12} {'Change':>12}")
print("-" * 50)
print(f"{'MAE':<10} {'$'+f'{baseline_mae:.2f}':>12} {'$'+f'{best_mae_final:.2f}':>12} {mae_imp:>+11.1f}%")
print(f"{'RMSE':<10} {'$'+f'{baseline_rmse:.2f}':>12} {'$'+f'{best_rmse_final:.2f}':>12} {rmse_imp:>+11.1f}%")
print(f"{'R²':<10} {baseline_r2:>12.4f} {best_r2_final:>12.4f} {r2_imp:>+11.1f}%")

print(f"\n🏆 Best Experiment: {best_name}")
print(f"⏱️  Total tuning time: {total_time/60:.1f} minutes")

# =============================================
#    STEP 7: BEST MODEL PARAMETERS
# =============================================
print("\n" + "=" * 60)
print("       BEST MODEL PARAMETERS")
print("=" * 60)

# Find best experiment params
for exp in experiments:
    if exp['name'] == best_name:
        for key, value in exp['params'].items():
            print(f"   {key}: {value}")
        break

# =============================================
#    STEP 8: SAVE EVERYTHING
# =============================================
os.makedirs("../models", exist_ok=True)

# Save best model
joblib.dump(best_model, "../models/fare_model_tuned.pkl")
print(f"\n✅ Best model saved → ../models/fare_model_tuned.pkl")

# Save results table
results_df.to_csv("../models/tuning_results.csv", index=False)
print(f"✅ Results table saved → ../models/tuning_results.csv")

# Save comparison
comparison = pd.DataFrame({
    'Model':    ['Baseline', 'Tuned'],
    'MAE':      [baseline_mae, best_mae_final],
    'RMSE':     [baseline_rmse, best_rmse_final],
    'R2':       [baseline_r2, best_r2_final],
})
comparison.to_csv("../models/model_comparison.csv", index=False)
print(f"✅ Comparison saved → ../models/model_comparison.csv")

print("\n" + "=" * 60)
print("       🎉 TUNING COMPLETE!")
print("=" * 60)