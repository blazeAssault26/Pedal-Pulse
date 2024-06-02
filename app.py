import streamlit as st
from predictor import Inference
import pandas as pd

# Load model and scaler
def load_model():
    model_path = "xgboost_reg_r2_0_944_v1.pkl"  
    sc_path = "sc.pkl"    
    inference = Inference(model_path, sc_path)
    return inference

inference = load_model()

def show_predict_page():
    st.title("Bike Rental Prediction")
    st.write("""### We need some information to predict the rented bike count""")

    date_input = st.date_input("Date", key="date")
    hour_input = st.slider("Hour", 0, 23, 12, key="hour")
    temperature_input = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0, step=0.1, key="temperature")
    humidity_input = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50, step=1, key="humidity")
    wind_speed_input = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key="wind_speed")
    visibility_input = st.number_input("Visibility (10m)", min_value=0.0, max_value=100.0, value=10.0, step=0.1, key="visibility")
    solar_radiation_input = st.number_input("Solar Radiation (MJ/m2)", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key="solar_radiation")
    rainfall_input = st.number_input("Rainfall (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="rainfall")
    snowfall_input = st.number_input("Snowfall (cm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="snowfall")
    seasons_input = st.selectbox("Season", ["Spring", "Summer", "Winter","Autumn"], key="season")
    holiday_input = st.selectbox("Holiday", ["No Holiday", "Holiday"], key="holiday")
    functioning_day_input = st.selectbox("Functioning Day", ["Yes", "No"], key="functioning_day")

    ok = st.button("Calculate Rental Count", key="ok_button")
    if ok:
        # Convert the date format to dd/mm/yyyy
        date_str = date_input.strftime("%d/%m/%Y")
        
        # Prepare the dataframe for prediction
        df = inference.prepare_dataframe(date_str, hour_input, temperature_input, humidity_input, wind_speed_input, visibility_input, solar_radiation_input, rainfall_input, snowfall_input, seasons_input, holiday_input, functioning_day_input)
        
        # Make prediction
        prediction = inference.predict(df)
        
        # Display the prediction
        st.subheader(f"The estimated rented bike count is {int(prediction[0])}")

show_predict_page()
