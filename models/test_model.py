# Cell 1 — Load model + data separately
import joblib
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split

# Load model
model = joblib.load("fare_model.pkl")

# Load data (same as model.py)
files = glob.glob("data/sampled/sample_*.parquet")
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

features = [
    "trip_miles", "trip_time", "PULocationID", "DOLocationID",
    "tips", "tolls", "congestion_surcharge", "airport_fee",
    "driver_pay", "trip_minutes", "trip_speed",
    "pickup_hour", "pickup_dayofweek", "is_weekend"
]
target = "base_passenger_fare"

X = df[features]
y = df[target]

# SAME random_state = same split!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
preds = model.predict(X_test)

# Cell 2 — Now all plots work
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, preds, alpha=0.3)
plt.xlabel("Actual Fare ($)")
plt.ylabel("Predicted Fare ($)")
plt.title("Actual vs Predicted Fare")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.savefig('visualizations/feature_importance.png', dpi=150)
plt.show()

##cel 3 
import seaborn as sns
import pandas as pd,matplotlib.pyplot as plt
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(importance['feature'], importance['importance'], color='steelblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance — What Drives Fare?')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=150)
plt.show()
print("✅ Feature Importance plot saved")