import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="FareFlow — Ride Fare Predictor",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #1a1a2e;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1.1rem;
            color: #555;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f4ff;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }
        .predict-result {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            margin-top: 1rem;
        }
        .stButton > button {
            width: 100%;
            background-color: #1a1a2e;
            color: white;
            border-radius: 10px;
            padding: 0.75rem;
            font-size: 1.1rem;
            font-weight: 600;
            border: none;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #16213e;
            color: #f0c040;
        }
        .info-box {
            background-color: #eef2ff;
            border-left: 5px solid #4f46e5;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "fare_model.pkl")
    
    # Try multiple paths for flexibility
    paths_to_try = [
        model_path,
        "models/fare_model.pkl",
        "../models/fare_model.pkl",
        "fare_model.pkl"
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            return joblib.load(path)
    
    st.error("❌ Model file not found. Please check your models folder.")
    st.stop()

model = load_model()


# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-title">🚖 FareFlow</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Scalable Ride Fare Prediction Engine — Powered by LightGBM</p>', unsafe_allow_html=True)
st.markdown("---")


# ============================================================
# LAYOUT — TWO COLUMNS
# ============================================================
left_col, right_col = st.columns([1, 1], gap="large")


# ============================================================
# LEFT COLUMN — INPUT FORM
# ============================================================
with left_col:
    st.markdown("### 📋 Trip Details")

    st.markdown("#### 🗺️ Trip Information")

    trip_miles = st.number_input(
        "Trip Distance (miles)",
        min_value=0.1,
        max_value=150.0,
        value=5.0,
        step=0.1,
        help="Total distance of the trip in miles"
    )

    trip_time = st.number_input(
        "Trip Duration (seconds)",
        min_value=60,
        max_value=10800,
        value=900,
        step=60,
        help="Total duration of the trip in seconds"
    )

    col1, col2 = st.columns(2)
    with col1:
        pu_location = st.number_input(
            "Pickup Zone ID",
            min_value=1,
            max_value=265,
            value=100,
            step=1,
            help="NYC Taxi Zone ID for pickup location"
        )
    with col2:
        do_location = st.number_input(
            "Dropoff Zone ID",
            min_value=1,
            max_value=265,
            value=150,
            step=1,
            help="NYC Taxi Zone ID for dropoff location"
        )

    st.markdown("#### 💰 Charges & Fees")

    col3, col4 = st.columns(2)
    with col3:
        tips = st.number_input(
            "Tips ($)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.5
        )
        tolls = st.number_input(
            "Tolls ($)",
            min_value=0.0,
            max_value=30.0,
            value=0.0,
            step=0.5
        )
    with col4:
        congestion_surcharge = st.number_input(
            "Congestion Surcharge ($)",
            min_value=0.0,
            max_value=10.0,
            value=2.75,
            step=0.25
        )
        airport_fee = st.number_input(
            "Airport Fee ($)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.5
        )

    driver_pay = st.number_input(
        "Driver Pay ($)",
        min_value=0.0,
        max_value=200.0,
        value=15.0,
        step=0.5,
        help="Estimated amount paid to the driver"
    )

    st.markdown("#### 🕐 Pickup Time")

    pickup_hour = st.slider(
        "Pickup Hour (0 = Midnight, 23 = 11PM)",
        min_value=0,
        max_value=23,
        value=12
    )

    pickup_day = st.selectbox(
        "Day of Week",
        options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    day_mapping = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    pickup_dayofweek = day_mapping[pickup_day]
    is_weekend = 1 if pickup_dayofweek >= 5 else 0

    # ---- PREDICT BUTTON ----
    st.markdown("---")
    predict_clicked = st.button("🚀 Predict Fare", type="primary")


# ============================================================
# RIGHT COLUMN — RESULTS
# ============================================================
with right_col:
    st.markdown("### 📊 Prediction Results")

    if predict_clicked:

        # ---- ENGINEERED FEATURES ----
        trip_minutes = trip_time / 60
        trip_speed = trip_miles / trip_minutes if trip_minutes > 0 else 0

        # ---- INPUT DATAFRAME ----
        input_data = pd.DataFrame([{
            'trip_miles': trip_miles,
            'trip_time': trip_time,
            'PULocationID': pu_location,
            'DOLocationID': do_location,
            'tips': tips,
            'tolls': tolls,
            'congestion_surcharge': congestion_surcharge,
            'airport_fee': airport_fee,
            'driver_pay': driver_pay,
            'trip_minutes': trip_minutes,
            'trip_speed': trip_speed,
            'pickup_hour': pickup_hour,
            'pickup_dayofweek': pickup_dayofweek,
            'is_weekend': is_weekend
        }])

        try:
            with st.spinner("⚡ Calculating fare..."):
                time.sleep(0.5)  # Smooth UX
                prediction = model.predict(input_data)[0]

            # ---- PREDICTION DISPLAY ----
            st.markdown(f"""
                <div class="predict-result">
                    <h2 style="color:#f0c040; margin-bottom:0.2rem;">Predicted Base Fare</h2>
                    <h1 style="font-size: 3.5rem; font-weight: 900; margin: 0;">${prediction:.2f}</h1>
                    <p style="color:#aaa; margin-top:0.5rem;">Based on your trip details</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # ---- TRIP STATS ----
            st.markdown("#### 🚗 Trip Statistics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Distance", f"{trip_miles:.1f} mi")
            c2.metric("Duration", f"{trip_minutes:.1f} min")
            c3.metric("Avg Speed", f"{trip_speed:.1f} mph")

            # ---- FARE BREAKDOWN ----
            st.markdown("#### 💵 Fare Breakdown")
            total_fare = prediction + tips + tolls + congestion_surcharge + airport_fee

            breakdown_df = pd.DataFrame({
                "Component": [
                    "Base Passenger Fare",
                    "Tips",
                    "Tolls",
                    "Congestion Surcharge",
                    "Airport Fee",
                    "💰 Total Estimated Cost"
                ],
                "Amount ($)": [
                    round(prediction, 2),
                    round(tips, 2),
                    round(tolls, 2),
                    round(congestion_surcharge, 2),
                    round(airport_fee, 2),
                    round(total_fare, 2)
                ]
            })

            st.dataframe(breakdown_df, width='stretch', hide_index=True)

            # ---- TOTAL HIGHLIGHT ----
            st.success(f"💰 **Total Estimated Trip Cost: ${total_fare:.2f}**")

            # ---- TIME CONTEXT ----
            st.markdown("#### 🕐 Time Context")
            t1, t2 = st.columns(2)
            t1.metric("Pickup Hour", f"{pickup_hour}:00")
            t2.metric("Day Type", "Weekend 🎉" if is_weekend else "Weekday 💼")

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Please check your input values and try again.")

    else:
        # ---- PLACEHOLDER WHEN NO PREDICTION ----
        st.markdown("""
            <div class="info-box">
                <h4>👈 How to use FareFlow</h4>
                <ol>
                    <li>Enter your trip distance and duration</li>
                    <li>Select pickup and dropoff zones</li>
                    <li>Add any charges or fees</li>
                    <li>Choose pickup time</li>
                    <li>Click <strong>Predict Fare</strong></li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

        # ---- MODEL INFO ----
        st.markdown("#### 🤖 Model Information")
        info_df = pd.DataFrame({
            "Detail": [
                "Algorithm",
                "Training Rows",
                "Dataset Size",
                "Features Used",
                "MAE",
                "RMSE",
                "R² Score"
            ],
            "Value": [
                "LightGBM Regressor",
                "1,000,000 rows",
                "20M records (sampled)",
                "14 features",
                "$4.03",
                "$7.33",
                "0.9051"
            ]
        })
        st.dataframe(info_df, width='stretch', hide_index=True)

        st.markdown("#### 📌 Features Used for Prediction")
        features_df = pd.DataFrame({
            "Feature": [
                "trip_miles", "trip_time", "PULocationID", "DOLocationID",
                "tips", "tolls", "congestion_surcharge", "airport_fee",
                "driver_pay", "trip_minutes", "trip_speed",
                "pickup_hour", "pickup_dayofweek", "is_weekend"
            ],
            "Type": [
                "Raw", "Raw", "Location", "Location",
                "Payment", "Payment", "Payment", "Payment",
                "Payment", "Engineered", "Engineered",
                "Time", "Time", "Time"
            ]
        })
        st.dataframe(features_df, width='stretch', hide_index=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
col_f1.markdown("🚖 **FareFlow** — Ride Fare Predictor")
col_f2.markdown("⚡ Powered by **LightGBM**")
col_f3.markdown("📦 Trained on **20M NYC Rides**")