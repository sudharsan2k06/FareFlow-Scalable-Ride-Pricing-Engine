import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================
#    GET PROJECT ROOT DIRECTORY
# =============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================
#    YOUR FOLDER PATHS — CHANGE HERE ONLY!
# =============================================
# Option 1: If you RENAMED the folder to "data"
DATA_FOLDER = "data"

# Option 2: If you kept original name (uncomment this, comment above)
# DATA_FOLDER = "Tuned&Comparison csv_files"

MODELS_FOLDER = "models"

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="NYC Fare Prediction",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM CSS ----
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-value { font-size: 36px; font-weight: bold; }
    .metric-label { font-size: 14px; opacity: 0.8; }
    .improvement-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# =============================================
#    DEBUG FUNCTION
# =============================================
def show_debug_info():
    """Show what files exist — helps debug deployment issues"""
    st.error("❌ Files not found!")
    st.write(f"**BASE_DIR:** `{BASE_DIR}`")
    
    st.write("---")
    st.write("**📁 All folders and files in project:**")
    
    for root, dirs, files in os.walk(BASE_DIR):
        # Skip hidden folders and venv
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv' and d != '__pycache__']
        
        level = root.replace(BASE_DIR, '').count(os.sep)
        indent = '  ' * level
        folder_name = os.path.basename(root) or 'ROOT'
        st.write(f"{indent}📂 **{folder_name}/**")
        
        subindent = '  ' * (level + 1)
        for file in files[:15]:
            st.write(f"{subindent}📄 {file}")
        if len(files) > 15:
            st.write(f"{subindent}*... and {len(files) - 15} more files*")
    
    st.stop()


# =============================================
#    CACHED FUNCTIONS — LOAD ONLY ONCE!
# =============================================

@st.cache_data(show_spinner="Loading dataset...")
def load_and_prepare_data():
    """Load data and engineer features — runs ONLY ONCE"""
    
    # ✅ YOUR DATA FOLDER
    data_dir = os.path.join(BASE_DIR, DATA_FOLDER)
    
    # Try to find parquet files
    parquet_pattern = os.path.join(data_dir, "*.parquet")
    files = glob.glob(parquet_pattern)
    
    # If not found, show debug info
    if len(files) == 0:
        st.error(f"❌ No .parquet files found in: `{data_dir}`")
        st.write(f"Looking for pattern: `{parquet_pattern}`")
        
        # Check if folder exists
        if os.path.exists(data_dir):
            st.write(f"✅ Folder exists: `{data_dir}`")
            st.write("Files in folder:")
            for f in os.listdir(data_dir)[:20]:
                st.write(f"  - {f}")
        else:
            st.write(f"❌ Folder does NOT exist: `{data_dir}`")
            show_debug_info()
        
        st.stop()
    
    st.sidebar.success(f"✅ Found {len(files)} data file(s)")
    
    # Load files (limit to 3 for memory on cloud)
    files = files[:3]
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # ---- Feature Engineering ----
    df['total_extras'] = df['tips'] + df['tolls'] + df['congestion_surcharge'] + df['airport_fee']
    df['fare_per_mile'] = df['driver_pay'] / df['trip_miles'].replace(0, np.nan)
    df['fare_per_minute'] = df['driver_pay'] / df['trip_minutes'].replace(0, np.nan)
    df['miles_per_minute'] = df['trip_miles'] / df['trip_minutes'].replace(0, np.nan)
    df['is_long_trip'] = (df['trip_miles'] > df['trip_miles'].quantile(0.75)).astype(int)
    df['is_short_trip'] = (df['trip_miles'] < df['trip_miles'].quantile(0.25)).astype(int)
    df['is_rush_hour'] = df['pickup_hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df['is_night'] = df['pickup_hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df['is_morning'] = df['pickup_hour'].isin([6, 7, 8, 9, 10, 11]).astype(int)

    def get_time_period(hour):
        if hour in [6, 7, 8, 9]: return 0
        elif hour in [10, 11, 12, 13, 14, 15]: return 1
        elif hour in [16, 17, 18, 19]: return 2
        elif hour in [20, 21, 22, 23]: return 3
        else: return 4

    df['time_period'] = df['pickup_hour'].apply(get_time_period)
    df['same_zone'] = (df['PULocationID'] == df['DOLocationID']).astype(int)

    airport_ids = [1, 132, 138]
    df['is_airport_pickup'] = df['PULocationID'].isin(airport_ids).astype(int)
    df['is_airport_dropoff'] = df['DOLocationID'].isin(airport_ids).astype(int)
    df['is_airport_trip'] = (df['is_airport_pickup'] | df['is_airport_dropoff']).astype(int)

    df['miles_x_speed'] = df['trip_miles'] * df['trip_speed']
    df['time_x_congestion'] = df['trip_minutes'] * df['congestion_surcharge']
    df['distance_x_hour'] = df['trip_miles'] * df['pickup_hour']

    df = df.fillna(0).replace([np.inf, -np.inf], 0)

    # ---- Outlier Removal ----
    q1 = df['base_passenger_fare'].quantile(0.01)
    q99 = df['base_passenger_fare'].quantile(0.99)
    df = df[(df['base_passenger_fare'] >= q1) & (df['base_passenger_fare'] <= q99)]
    df = df[df['base_passenger_fare'] > 0]
    df = df[df['trip_miles'] > 0]
    df = df[df['trip_miles'] < df['trip_miles'].quantile(0.99)]
    df = df[df['trip_time'] > 0]
    df = df[df['trip_time'] < df['trip_time'].quantile(0.99)]

    return df


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load trained models — runs ONLY ONCE"""
    
    # ✅ YOUR MODELS FOLDER
    models_dir = os.path.join(BASE_DIR, MODELS_FOLDER)
    
    baseline_path = os.path.join(models_dir, "fare_model.pkl")
    tuned_path = os.path.join(models_dir, "fare_model_tuned.pkl")
    improved_path = os.path.join(models_dir, "fare_model_improved.pkl")
    
    # Check if models exist
    missing = []
    for path, name in [(baseline_path, "Baseline"), (tuned_path, "Tuned"), (improved_path, "Improved")]:
        if not os.path.exists(path):
            missing.append((name, path))
    
    if missing:
        st.error("❌ Models not found!")
        for name, path in missing:
            st.write(f"- **{name}**: `{path}`")
        
        st.write("---")
        st.write(f"**Models folder:** `{models_dir}`")
        
        if os.path.exists(models_dir):
            st.write("**Files in models folder:**")
            for f in os.listdir(models_dir):
                st.write(f"  - {f}")
        else:
            st.write("❌ Models folder does NOT exist!")
            show_debug_info()
        
        st.stop()
    
    baseline = joblib.load(baseline_path)
    tuned = joblib.load(tuned_path)
    improved = joblib.load(improved_path)
    
    st.sidebar.success("✅ All models loaded")
    return baseline, tuned, improved


@st.cache_data(show_spinner="Generating predictions...")
def prepare_predictions(_baseline, _tuned, _improved, X_test_df, y_test_series, original_features):
    """Generate predictions — runs ONLY ONCE"""
    baseline_preds = _baseline.predict(X_test_df[original_features])
    tuned_preds = _tuned.predict(X_test_df[original_features])
    improved_preds = _improved.predict(X_test_df)
    return baseline_preds, tuned_preds, improved_preds


# =============================================
#    LOAD EVERYTHING
# =============================================
df = load_and_prepare_data()
baseline_model, tuned_model, improved_model = load_models()

# ---- Feature Lists ----
original_features = [
    "trip_miles", "trip_time", "PULocationID", "DOLocationID",
    "tips", "tolls", "congestion_surcharge", "airport_fee",
    "driver_pay", "trip_minutes", "trip_speed",
    "pickup_hour", "pickup_dayofweek", "is_weekend"
]

new_features = [
    "total_extras", "fare_per_mile", "fare_per_minute",
    "miles_per_minute", "is_long_trip", "is_short_trip",
    "is_rush_hour", "is_night", "is_morning", "time_period",
    "same_zone", "is_airport_pickup", "is_airport_dropoff",
    "is_airport_trip", "miles_x_speed", "time_x_congestion",
    "distance_x_hour"
]

all_features = original_features + new_features
target = "base_passenger_fare"

X = df[all_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Cached Predictions ----
baseline_preds, tuned_preds, improved_preds = prepare_predictions(
    baseline_model, tuned_model, improved_model, X_test, y_test, original_features
)

# ---- Calculate Metrics ----
metrics = {}
for name, preds in [('Baseline', baseline_preds), ('Tuned', tuned_preds), ('Improved', improved_preds)]:
    metrics[name] = {
        'MAE': mean_absolute_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'R2': r2_score(y_test, preds)
    }


# =============================================
#              SIDEBAR
# =============================================
st.sidebar.title("🚕 NYC Fare Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview",
     "📊 Data Exploration",
     "🤖 Model Comparison",
     "🔮 Predict Fare",
     "📈 Feature Analysis",
     "📋 Project Report"]
)

st.sidebar.markdown("---")
st.sidebar.success(f"""
**Dataset:** {len(df):,} rides  
**Features:** {len(all_features)}  
**Models:** 3
""")


# =============================================
#         PAGE 1: OVERVIEW
# =============================================
if page == "🏠 Overview":

    st.markdown("<h1 style='text-align:center;'>🚕 NYC Ride Fare Prediction</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px; color:gray;'>"
                "ML Pipeline: Baseline → Tuning → Feature Engineering</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Rides</div>
            <div class='metric-value'>{len(df):,}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card' style='background:linear-gradient(135deg,#f093fb,#f5576c);'>
            <div class='metric-label'>Best MAE</div>
            <div class='metric-value'>${metrics["Improved"]["MAE"]:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card' style='background:linear-gradient(135deg,#4facfe,#00f2fe);'>
            <div class='metric-label'>Best RMSE</div>
            <div class='metric-value'>${metrics["Improved"]["RMSE"]:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        improvement = ((metrics['Improved']['MAE'] - metrics['Baseline']['MAE'])
                       / metrics['Baseline']['MAE'] * 100)
        st.markdown(f"""
        <div class='improvement-card'>
            <div class='metric-label'>Improvement</div>
            <div class='metric-value'>{improvement:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📍 Model Improvement Journey")
    journey = pd.DataFrame({
        'Stage': ['1. Baseline', '2. Tuned', '3. Improved'],
        'MAE ($)': [metrics['Baseline']['MAE'], metrics['Tuned']['MAE'], metrics['Improved']['MAE']],
        'RMSE ($)': [metrics['Baseline']['RMSE'], metrics['Tuned']['RMSE'], metrics['Improved']['RMSE']],
        'R²': [metrics['Baseline']['R2'], metrics['Tuned']['R2'], metrics['Improved']['R2']],
    })
    st.dataframe(journey.style.format({'MAE ($)': '${:.2f}', 'RMSE ($)': '${:.2f}', 'R²': '{:.4f}'}),
                 use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=['Baseline', 'Tuned', 'Improved'],
        y=[metrics['Baseline']['MAE'], metrics['Tuned']['MAE'], metrics['Improved']['MAE']],
        mode='lines+markers+text',
        text=[f"${v:.2f}" for v in [metrics['Baseline']['MAE'], metrics['Tuned']['MAE'], metrics['Improved']['MAE']]],
        textposition='top center',
        line=dict(color='#667eea', width=3),
        marker=dict(size=15)
    ))
    fig.update_layout(title="MAE Improvement Journey", yaxis_title="MAE ($)", height=400)
    st.plotly_chart(fig, use_container_width=True)


# =============================================
#         PAGE 2: DATA EXPLORATION
# =============================================
elif page == "📊 Data Exploration":

    st.title("📊 Data Exploration")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Target", "📦 Features", "🔥 Correlations", "📋 Data"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=target, nbins=100, title="Fare Distribution",
                               color_discrete_sequence=['#667eea'])
            fig.add_vline(x=y.mean(), line_dash="dash", line_color="red",
                          annotation_text=f"Mean: ${y.mean():.2f}")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, y=target, title="Fare Box Plot",
                         color_discrete_sequence=['#f5576c'])
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        selected = st.selectbox("Select Feature:", all_features)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=selected, nbins=80, title=f"{selected} Distribution",
                               color_discrete_sequence=['teal'])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, y=selected, title=f"{selected} Box Plot",
                         color_discrete_sequence=['coral'])
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        num_cols = ["trip_miles", "trip_time", "driver_pay", "tips",
                    "trip_speed", "fare_per_mile", "fare_per_minute",
                    "total_extras", target]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Correlation Matrix", aspect="auto")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.dataframe(df.head(100), use_container_width=True)


# =============================================
#         PAGE 3: MODEL COMPARISON
# =============================================
elif page == "🤖 Model Comparison":

    st.title("🤖 Model Comparison")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    for col, (name, color) in zip(
        [col1, col2, col3],
        [('Baseline', '🔴'), ('Tuned', '🟡'), ('Improved', '🟢')]
    ):
        with col:
            st.subheader(f"{color} {name}")
            st.metric("MAE", f"${metrics[name]['MAE']:.2f}")
            st.metric("RMSE", f"${metrics[name]['RMSE']:.2f}")
            st.metric("R²", f"{metrics[name]['R2']:.4f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Actual vs Predicted", "📉 Residuals",
        "📊 Error Distribution", "🏆 Feature Importance"
    ])

    with tab1:
        sample_size = st.slider("Sample size:", 1000, 20000, 5000)
        idx = np.random.choice(len(y_test), min(sample_size, len(y_test)), replace=False)

        fig = go.Figure()
        for preds, name, color in [
            (baseline_preds, 'Baseline', 'red'),
            (tuned_preds, 'Tuned', 'orange'),
            (improved_preds, 'Improved', 'green')
        ]:
            fig.add_trace(go.Scatter(
                x=y_test.values[idx], y=preds[idx],
                mode='markers', name=name,
                marker=dict(color=color, size=3, opacity=0.3)
            ))
        min_v, max_v = y_test.values[idx].min(), y_test.values[idx].max()
        fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                 mode='lines', name='Perfect',
                                 line=dict(color='black', dash='dash')))
        fig.update_layout(title="Actual vs Predicted", height=600,
                          xaxis_title="Actual ($)", yaxis_title="Predicted ($)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        for preds, name, color in [
            (baseline_preds, 'Baseline', 'red'),
            (tuned_preds, 'Tuned', 'orange'),
            (improved_preds, 'Improved', 'green')
        ]:
            residuals = y_test.values - preds
            fig.add_trace(go.Scatter(x=preds[::10], y=residuals[::10],
                                     mode='markers', name=name,
                                     marker=dict(color=color, size=3, opacity=0.2)))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(title="Residual Plot", height=500,
                          xaxis_title="Predicted ($)", yaxis_title="Residual ($)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure()
        for preds, name, color in [
            (baseline_preds, 'Baseline', 'red'),
            (tuned_preds, 'Tuned', 'orange'),
            (improved_preds, 'Improved', 'green')
        ]:
            errors = np.abs(y_test.values - preds)
            fig.add_trace(go.Histogram(x=errors, name=f"{name} (${np.mean(errors):.2f})",
                                       opacity=0.5, nbinsx=100, marker_color=color))
        fig.update_layout(title="Error Distribution", barmode='overlay',
                          xaxis=dict(range=[0, 25]), xaxis_title="Absolute Error ($)", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        importance = pd.DataFrame({
            'Feature': all_features,
            'Importance': improved_model.feature_importances_,
            'Type': ['🔵 Original' if f in original_features else '🟢 New' for f in all_features]
        }).sort_values('Importance', ascending=True)
        fig = px.bar(importance, x='Importance', y='Feature', color='Type',
                     orientation='h', title="Feature Importance",
                     color_discrete_map={'🔵 Original': '#3498db', '🟢 New': '#27ae60'})
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)


# =============================================
#    PAGE 4: PREDICT FARE
# =============================================
elif page == "🔮 Predict Fare":

    st.title("🔮 Predict Your Fare")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🚗 Trip Details")
        trip_miles = st.number_input("Trip Miles", 0.1, 100.0, 5.0, 0.5)
        trip_time = st.number_input("Trip Time (seconds)", 60, 7200, 900, 60)
        trip_minutes = trip_time / 60
        trip_speed = (trip_miles / trip_minutes * 60) if trip_minutes > 0 else 0
        st.info(f"⏱️ {trip_minutes:.1f} min | 🏎️ {trip_speed:.1f} mph")

    with col2:
        st.subheader("📍 Location & Time")
        pu_location = st.number_input("Pickup Location ID", 1, 265, 132)
        do_location = st.number_input("Dropoff Location ID", 1, 265, 68)
        pickup_hour = st.slider("Pickup Hour", 0, 23, 14)
        pickup_day = st.selectbox("Day",
                                  ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                   'Friday', 'Saturday', 'Sunday'])
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
                   'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        pickup_dayofweek = day_map[pickup_day]
        is_weekend = 1 if pickup_dayofweek >= 5 else 0

    with col3:
        st.subheader("💰 Cost Components")
        tips = st.number_input("Tips ($)", 0.0, 50.0, 2.0, 0.5)
        tolls = st.number_input("Tolls ($)", 0.0, 30.0, 0.0, 0.5)
        congestion = st.number_input("Congestion ($)", 0.0, 5.0, 2.75, 0.25)
        airport_fee = st.number_input("Airport Fee ($)", 0.0, 5.0, 0.0, 0.25)
        driver_pay = st.number_input("Driver Pay ($)", 0.0, 200.0, 15.0, 1.0)

    st.markdown("---")

    if st.button("🚕 PREDICT FARE", use_container_width=True, type="primary"):

        # Calculate engineered features
        total_extras = tips + tolls + congestion + airport_fee
        fare_per_mile = driver_pay / trip_miles if trip_miles > 0 else 0
        fare_per_minute = driver_pay / trip_minutes if trip_minutes > 0 else 0
        miles_per_minute = trip_miles / trip_minutes if trip_minutes > 0 else 0
        is_long_trip = 1 if trip_miles > 8 else 0
        is_short_trip = 1 if trip_miles < 2 else 0
        is_rush_hour = 1 if pickup_hour in [7, 8, 9, 16, 17, 18, 19] else 0
        is_night = 1 if pickup_hour in [22, 23, 0, 1, 2, 3, 4, 5] else 0
        is_morning = 1 if pickup_hour in [6, 7, 8, 9, 10, 11] else 0

        if pickup_hour in [6, 7, 8, 9]: time_period = 0
        elif pickup_hour in [10, 11, 12, 13, 14, 15]: time_period = 1
        elif pickup_hour in [16, 17, 18, 19]: time_period = 2
        elif pickup_hour in [20, 21, 22, 23]: time_period = 3
        else: time_period = 4

        same_zone = 1 if pu_location == do_location else 0
        airport_ids_list = [1, 132, 138]
        is_airport_pickup = 1 if pu_location in airport_ids_list else 0
        is_airport_dropoff = 1 if do_location in airport_ids_list else 0
        is_airport_trip = 1 if is_airport_pickup or is_airport_dropoff else 0
        miles_x_speed = trip_miles * trip_speed
        time_x_congestion = trip_minutes * congestion
        distance_x_hour = trip_miles * pickup_hour

        input_data = pd.DataFrame([[
            trip_miles, trip_time, pu_location, do_location,
            tips, tolls, congestion, airport_fee,
            driver_pay, trip_minutes, trip_speed,
            pickup_hour, pickup_dayofweek, is_weekend,
            total_extras, fare_per_mile, fare_per_minute,
            miles_per_minute, is_long_trip, is_short_trip,
            is_rush_hour, is_night, is_morning, time_period,
            same_zone, is_airport_pickup, is_airport_dropoff,
            is_airport_trip, miles_x_speed, time_x_congestion,
            distance_x_hour
        ]], columns=all_features)

        pred_baseline = baseline_model.predict(input_data[original_features])[0]
        pred_tuned = tuned_model.predict(input_data[original_features])[0]
        pred_improved = improved_model.predict(input_data)[0]

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class='prediction-card'>
                <h2>🏆 Best Model Prediction</h2>
                <h1 style='font-size:64px;'>${pred_improved:.2f}</h1>
                <p>Estimated base passenger fare</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 Baseline", f"${pred_baseline:.2f}")
        col2.metric("🟡 Tuned", f"${pred_tuned:.2f}")
        col3.metric("🟢 Improved", f"${pred_improved:.2f}")


# =============================================
#    PAGE 5: FEATURE ANALYSIS
# =============================================
elif page == "📈 Feature Analysis":

    st.title("📈 Feature Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔍 Feature vs Fare", "⏰ Time Patterns", "📊 Stats"])

    # ---- TAB 1: Feature vs Fare ----
    with tab1:
        st.subheader("🔍 Feature vs Fare Relationship")
        
        selected = st.selectbox("Select Feature:", all_features)
        sample_df = df.sample(min(10000, len(df)), random_state=42)
        
        # ✅ FIXED: Removed trendline="ols"
        fig = px.scatter(
            sample_df, 
            x=selected, 
            y=target, 
            opacity=0.3,
            color_discrete_sequence=['teal'],
            title=f"{selected} vs Fare"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation
        if selected in df.columns and target in df.columns:
            corr = df[selected].corr(df[target])
            st.info(f"📊 Correlation between **{selected}** and **{target}**: **{corr:.4f}**")

    # ---- TAB 2: Time Patterns ----
    with tab2:
        st.subheader("⏰ Time-Based Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly pattern
            if 'pickup_hour' in df.columns:
                hourly = df.groupby('pickup_hour')[target].mean().reset_index()
                fig = px.bar(
                    hourly, 
                    x='pickup_hour', 
                    y=target, 
                    title="Average Fare by Hour",
                    color=target, 
                    color_continuous_scale='teal'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("pickup_hour column not found")
        
        with col2:
            # Daily pattern
            if 'pickup_dayofweek' in df.columns:
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                daily = df.groupby('pickup_dayofweek')[target].mean().reset_index()
                
                # Safe day name mapping
                daily['day'] = daily['pickup_dayofweek'].apply(
                    lambda x: days[int(x)] if 0 <= int(x) < 7 else 'Unknown'
                )
                
                fig = px.bar(
                    daily, 
                    x='day', 
                    y=target, 
                    title="Average Fare by Day",
                    color=target, 
                    color_continuous_scale='sunset'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("pickup_dayofweek column not found")
        
        # Rush hour vs Normal
        col1, col2 = st.columns(2)
        
        with col1:
            if 'is_rush_hour' in df.columns:
                rush = df.groupby('is_rush_hour')[target].mean().reset_index()
                rush['Type'] = rush['is_rush_hour'].map({0: 'Normal', 1: 'Rush Hour'})
                fig = px.bar(
                    rush, 
                    x='Type', 
                    y=target, 
                    color='Type',
                    title="Rush Hour vs Normal",
                    color_discrete_sequence=['steelblue', 'coral']
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'is_airport_trip' in df.columns:
                airport = df.groupby('is_airport_trip')[target].mean().reset_index()
                airport['Type'] = airport['is_airport_trip'].map({0: 'Regular', 1: 'Airport'})
                fig = px.bar(
                    airport, 
                    x='Type', 
                    y=target, 
                    color='Type',
                    title="Airport vs Regular Trips",
                    color_discrete_sequence=['steelblue', 'gold']
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 3: Stats ----
    with tab3:
        st.subheader("📊 Feature Statistics")
        
        # Get numeric columns only
        numeric_cols = df[all_features + [target]].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe().T
            
            # Format nicely
            st.dataframe(
                stats_df.style.format("{:.2f}").background_gradient(cmap='Blues', axis=1),
                use_container_width=True,
                height=600
            )
        else:
            st.warning("No numeric columns found for statistics")
        
        st.markdown("---")
        
        # Missing values
        st.subheader("🔍 Missing Values")
        missing = df[all_features + [target]].isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        }).sort_values('Missing %', ascending=False)
        
        if missing_df['Missing Count'].sum() > 0:
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        else:
            st.success("✅ No missing values!")

# =============================================
#         PAGE 6: PROJECT REPORT
# =============================================
elif page == "📋 Project Report":

    st.title("📋 Project Report")
    st.markdown("---")

    st.markdown(f"""
    ## 🚕 NYC Ride Fare Prediction — Summary

    ### Dataset
    - **Total Rides:** {len(df):,}
    - **Features:** {len(all_features)} (14 original + 17 engineered)
    - **Target:** base_passenger_fare

    ### Model Evolution

    | Stage | MAE | RMSE | R² |
    |-------|-----|------|----|
    | Baseline | ${metrics['Baseline']['MAE']:.2f} | ${metrics['Baseline']['RMSE']:.2f} | {metrics['Baseline']['R2']:.4f} |
    | Tuned | ${metrics['Tuned']['MAE']:.2f} | ${metrics['Tuned']['RMSE']:.2f} | {metrics['Tuned']['R2']:.4f} |
    | Improved | ${metrics['Improved']['MAE']:.2f} | ${metrics['Improved']['RMSE']:.2f} | {metrics['Improved']['R2']:.4f} |

    ### Key Findings
    1. **driver_pay** is the strongest predictor
    2. **fare_per_minute** (engineered) ranked #2
    3. **Feature engineering** > parameter tuning
    """)

    importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': improved_model.feature_importances_,
        'Type': ['Original' if f in original_features else '⭐ New' for f in all_features]
    }).sort_values('Importance', ascending=False).head(10)
    st.subheader("🏆 Top 10 Features")
    st.dataframe(importance, use_container_width=True)


# ---- Footer ----
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ Streamlit")