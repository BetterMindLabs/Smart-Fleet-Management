import streamlit as st
import pandas as pd
import google.generativeai as genai
from ml_models import predict_maintenance, predict_fuel

# -------------------------------
# Configure Gemini
# -------------------------------
genai.configure(api_key=st.secrets["API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

st.title("🚚 Smart Fleet Management with ML & Gemini")

st.markdown("Intelligently predict maintenance, estimate fuel, and get AI-driven insights.")

# --- User input
vehicle_id = st.text_input("Vehicle ID", "Truck-01")
km_since_maintenance = st.number_input("Km since last maintenance", min_value=0, value=12000)
avg_daily_km = st.number_input("Average daily km", min_value=0, value=100)
last_maintenance_days = st.number_input("Days since last maintenance", min_value=0, value=180)
fuel_consumption = st.number_input("Current fuel consumption (L/100km)", min_value=0.0, value=30.0)

if st.button("Analyze Fleet Data"):
    with st.spinner("Running ML predictions and AI analysis..."):
        # ML predictions
        maintenance_needed = predict_maintenance(km_since_maintenance, avg_daily_km, last_maintenance_days, fuel_consumption)
        predicted_fuel = predict_fuel(km_since_maintenance, avg_daily_km, last_maintenance_days)
        
        # Gemini advice
        prompt = f"""
        Vehicle ID: {vehicle_id}
        Km since maintenance: {km_since_maintenance}
        Average daily km: {avg_daily_km}
        Days since last maintenance: {last_maintenance_days}
        Current fuel consumption: {fuel_consumption}

        Maintenance needed (ML prediction): {bool(maintenance_needed)}
        Predicted fuel consumption next period: {predicted_fuel:.2f} L/100km.

        Suggest:
        - Maintenance actions and urgency.
        - Driving behavior improvements to reduce fuel consumption.
        - Emission impact and tips to improve sustainability.
        Give a clear, friendly bullet-point summary.
        """

        response = model.generate_content(prompt)
        advice = response.text

        st.success("✅ ML Predictions & Gemini Recommendations")
        st.markdown(f"**Maintenance Needed?** {'Yes' if maintenance_needed else 'No'}")
        st.markdown(f"**Estimated Fuel Consumption:** {predicted_fuel:.2f} L/100km")
        st.markdown("---")
        st.markdown(advice)

st.caption("♻️ Built with ML, Gemini API, and Streamlit")
