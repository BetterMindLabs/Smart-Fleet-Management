import streamlit as st
import pandas as pd
import google.generativeai as genai

# ----------------------
# CONFIGURE GEMINI API
# ----------------------
genai.configure(api_key=st.secrets["API_KEY"])  
model = genai.GenerativeModel("gemini-2.5-flash")

# ----------------------
# STREAMLIT UI
# ----------------------
st.title("🚚 Smart Fleet Management System")

st.markdown("""
Manage your fleet intelligently — optimize routes, reduce maintenance costs, and lower emissions using AI.
""")

# --- Vehicle Data Entry
with st.form("vehicle_form"):
    vehicle_id = st.text_input("Vehicle ID or Name", "Truck-01")
    vehicle_type = st.selectbox("Vehicle Type", ["Truck", "Van", "Bus", "Car"])
    fuel_type = st.selectbox("Fuel Type", ["Diesel", "Petrol", "Electric", "Hybrid"])
    last_maintenance_km = st.number_input("Km since last maintenance", min_value=0, max_value=100000, value=5000)
    avg_daily_km = st.number_input("Average daily distance (km)", min_value=0, max_value=1000, value=100)
    current_location = st.text_input("Current Location", "Warehouse A")
    submit = st.form_submit_button("Analyze & Get Advice")

if submit:
    with st.spinner("Analyzing fleet data..."):

        # Prepare prompt for Gemini
        prompt = f"""
        Act as an expert fleet management AI assistant.

        Vehicle Details:
        - ID: {vehicle_id}
        - Type: {vehicle_type}
        - Fuel: {fuel_type}
        - Km since last maintenance: {last_maintenance_km}
        - Average daily distance: {avg_daily_km}
        - Current location: {current_location}

        Please suggest:
        - Maintenance recommendation and urgency level.
        - Fuel or energy consumption optimization tips.
        - Emission impact and ways to reduce it.
        - Route optimization suggestions for cost and efficiency.
        Present these in a clear, bullet-point summary.
        """

        response = model.generate_content(prompt)
        advice = response.text

        st.success("✅ AI Recommendations")
        st.markdown(advice)

# --- Simulated Fleet Table (optional demo)
st.markdown("### 📊 Sample Fleet Overview")
fleet_data = pd.DataFrame({
    "Vehicle ID": ["Truck-01", "Van-03", "Bus-05"],
    "Type": ["Truck", "Van", "Bus"],
    "Fuel": ["Diesel", "Petrol", "Electric"],
    "Km since last maintenance": [5000, 12000, 3000],
    "Avg daily km": [100, 80, 60],
    "Location": ["Warehouse A", "Depot B", "Garage C"],
})
st.dataframe(fleet_data)

st.caption("♻️ Powered by Gemini API, Streamlit & Python")
