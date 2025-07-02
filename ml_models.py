import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("data.csv")

# -------------------------------
# Maintenance Prediction
# -------------------------------
def train_maintenance_model():
    X = df[["km_since_maintenance", "avg_daily_km", "last_maintenance_days", "fuel_consumption_l_per_100km"]]
    y = df["needs_maintenance"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# -------------------------------
# Fuel Consumption Prediction
# -------------------------------
def train_fuel_model():
    X = df[["km_since_maintenance", "avg_daily_km", "last_maintenance_days"]]
    y = df["fuel_consumption_l_per_100km"]
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return model

# -------------------------------
# Predict functions
# -------------------------------
maintenance_model = train_maintenance_model()
fuel_model = train_fuel_model()

def predict_maintenance(km, daily_km, last_days, fuel_cons):
    X_new = [[km, daily_km, last_days, fuel_cons]]
    return maintenance_model.predict(X_new)[0]

def predict_fuel(km, daily_km, last_days):
    X_new = [[km, daily_km, last_days]]
    return fuel_model.predict(X_new)[0]
