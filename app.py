import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Traffic AI Optimizer", layout="centered")

# --- MODEL LOADING ---
@st.cache_resource
def load_model_assets():
    # If the model doesn't exist yet, this script will notify you
    if not os.path.exists('traffic_model.pkl'):
        st.error("Model files not found. Please run your training script first!")
        return None, None
    model = joblib.load('traffic_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

model, model_columns = load_model_assets()

# --- UI DESIGN ---
st.title("🚦 Traffic Light AI Optimizer")
st.markdown("Enter real-time traffic data to calculate the optimal green light duration.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        vehicle_count = st.number_input("Vehicle Count", min_value=0, value=20)
        avg_speed = st.slider("Average Speed (km/h)", 10, 100, 40)
        lane_id = st.selectbox("Lane Direction", ["North_S", "North_L", "East_S", "East_L", "South_S", "South_L", "West_S", "West_L"])
    
    with col2:
        hour = st.slider("Hour of Day", 0, 23, 12)
        weather = st.selectbox("Weather", ["Clear", "Rainy", "Cloudy", "Foggy"])
        ambulance = st.toggle("Ambulance Present?")

    submit = st.form_submit_button("Calculate Green Time")

# --- PREDICTION LOGIC ---
if submit and model is not None:
    # 1. Create input dictionary
    input_data = {
        'Hour': hour,
        'Lane_Length_m': 200, # Default value
        'Vehicle_Count': vehicle_count,
        'Avg_Speed_kmh': avg_speed,
        'Ambulance_Present': 1 if ambulance else 0,
        f'Lane_ID_{lane_id}': 1,
        f'Weather_{weather}': 1
    }
    
    # 2. Convert to DataFrame and align with training columns
    query_df = pd.DataFrame([input_data])
    for col in model_columns:
        if col not in query_df.columns:
            query_df[col] = 0
            
    query_df = query_df[model_columns] # Reorder
    
    # 3. Predict
    prediction = model.predict(query_df)[0]
    
    st.success(f"### Recommended Green Time: {round(prediction, 2)} Seconds")
    st.progress(min(int(prediction), 100))