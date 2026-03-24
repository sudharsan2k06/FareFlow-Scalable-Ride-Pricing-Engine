<h1 align="center">🚕 NYC Taxi Fare Prediction</h1>

<p align="center">
  <em>A complete Machine Learning pipeline that processes 20M+ NYC taxi records 
  and predicts passenger fares with ~85% accuracy using LightGBM.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-Regressor-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Data-20M%2B%20Records-orange?style=for-the-badge" />
</p>

---

## 📌 Overview

This project builds an **end-to-end ML pipeline** to predict NYC taxi `base_passenger_fare` using trip details like distance, time, location, and cost components.

- 📦 Processed **20M+ raw trip records**, sampled **1M for training**
- 🧠 Engineered **17 new features** on top of 14 original features
- 📈 Improved prediction error by **14.7%** through feature engineering
- 🖥️ Built interactive **Streamlit dashboard** for real-time predictions

---

## 📸 Screenshots

### 🏠 Dashboard Overview

<p align="center">
  <img width="1916" height="973" alt="Screenshot 2026-03-18 005543" src="https://github.com/user-attachments/assets/c72202f8-7d7c-4415-ad13-e313dc7ec7f8" />

  <br />
  <em>Key metrics and model improvement journey</em>
</p>

---

### 📊 Data Exploration

<p align="center">
  <img width="1887" height="866" alt="Screenshot 2026-03-18 010709" src="https://github.com/user-attachments/assets/5f56b13f-5116-457e-8b42-f4d7718521cb" />

  <br />
  <em>Target distribution, feature analysis, and correlation heatmap</em>
</p>


---

### 🔮 Real-Time Fare Prediction

<p align="center">
  <img width="1916" height="916" alt="Screenshot 2026-03-18 005730" src="https://github.com/user-attachments/assets/b697cfab-ee19-4d28-97a8-d0129d8da299" />

  <br />
  <em>Input trip details and get instant fare prediction from all 3 models</em>
</p>

---

### 📈 Feature Importance

<p align="center">
  <img width="1904" height="863" alt="Screenshot 2026-03-18 010338" src="https://github.com/user-attachments/assets/b85622c2-5c28-470a-ac6d-9675dee25d48" />

  <br />
  <em>Green = Engineered features | Blue = Original features</em>
</p>

---

---

## 🧠 Model Details

| Detail | Value |
|--------|-------|
| **Algorithm** | LightGBM Regressor |
| **Problem Type** | Regression |
| **Target Variable** | `base_passenger_fare` |
| **Raw Dataset** | 20M+ records |
| **Training Dataset** | 1M sampled records |
| **Total Features** | 31 (14 original + 17 engineered) |

---

## 📊 Model Improvement Journey

| Stage | MAE ($) | RMSE ($) | R² | What Changed |
|-------|---------|----------|----|-------------|
| 🔴 Baseline | 4.03 | 7.33 | 0.9051 | 14 features, default params |
| 🟡 Tuned | 3.98 | 7.28 | 0.9064 | Hyperparameter tuning |
| 🟢 Improved | 3.44 | 5.39 | 0.8958 | 31 features + outlier removal |

> **Total MAE Improvement: -14.7% ($4.03 → $3.44)**
> 
> **Total RMSE Improvement: -26.5% ($7.33 → $5.39)**

---

## 🔧 Engineered Features

| Feature | Description | Importance Rank |
|---------|-------------|-----------------|
| `fare_per_minute` | Driver pay ÷ trip minutes | ⭐ #2 |
| `fare_per_mile` | Driver pay ÷ trip miles | ⭐ #4 |
| `distance_x_hour` | Trip miles × pickup hour | ⭐ #6 |
| `time_x_congestion` | Trip minutes × congestion surcharge | ⭐ #9 |
| `total_extras` | Sum of tips, tolls, congestion, airport fee | ⭐ #10 |
| `miles_x_speed` | Trip miles × trip speed | ⭐ #11 |
| `is_rush_hour` | 1 if pickup during rush hours (7-9, 16-19) | — |
| `is_airport_trip` | 1 if pickup/dropoff at airport | — |
| `is_night` | 1 if pickup during night hours (22-5) | — |
| `same_zone` | 1 if pickup = dropoff zone | — |
| `is_long_trip` | 1 if trip miles > 75th percentile | — |

> **6 out of the top 11 most important features were newly engineered ⭐**

---

## 📈 Top 10 Features by Importance

driver_pay ██████████████████████████████

fare_per_minute █████████████████████████████ ⭐ NEW

PULocationID ██████████████████████████

fare_per_mile ██████████████████████████ ⭐ NEW

DOLocationID ████████████████████████

distance_x_hour █████████████████████ ⭐ NEW

trip_time █████████████████████

trip_miles ███████████████████

time_x_congestion ███████████████ ⭐ NEW

total_extras ███████████████ ⭐ NEW

---

## 🏗️ Project Structure

nyc_ML_model/

│

├── data/ # Dataset files

│ └── sampled/

│ ├── sample_1.parquet

│ ├── sample_2.parquet

│ └── ...

│

├── models/ # Training scripts

│ ├── train_model.py # Step 1: Baseline model

│ ├── tune_model.py # Step 2: Hyperparameter tuning

│ └── improve_model.py # Step 3: Feature engineering

│

├── saved_models/ # Trained model files

│ ├── fare_model.pkl # Baseline model

│ ├── fare_model_tuned.pkl # Tuned model

│ └── fare_model_improved.pkl # Final improved model

│

├── results/ # Experiment results

│ ├── tuning_results.csv

│ ├── improvement_results.csv

│ └── full_comparison.csv

│

├── notebooks/ # Jupyter notebooks

│ ├── 01_EDA.ipynb # Exploratory Data Analysis

│ └── 02_post_tuning_visuals.ipynb # Model comparison visuals

│

├── app/ # Streamlit dashboard

│ └── app.py

│

├── screenshots/ # README images

│ ├── overview.png

│ ├── eda.png

│ ├── model_comparison.png

│ ├── predict_fare.png

│ ├── feature_importance.png

│ └── report.png

│

├── requirements.txt # Dependencies

├── .gitignore # Git ignore rules

└── README.md # This file

---
## HOW IT WORKS !!

User Inputs  &  Model Predicts

─────────────  ───────────────

🚗 Trip distance & time              

📍 Pickup & dropoff locations     →   💰 Estimated Fare

⏰ Time of day & day of week          in USD

💰 Tips, tolls, congestion            

##👨‍💻 Author

SUDHARRSAN G

LinkedIn---> https://www.linkedin.com/in/sudharsan-g-4b11b6348/?skipRedirect=true

