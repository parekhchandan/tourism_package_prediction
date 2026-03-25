import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load model
model_path = hf_hub_download(
    repo_id="Chandan2312/tourism-package-prediction",
    filename="best_customer_model_v1.joblib"
)
model = joblib.load(model_path)

# 🎉 Attractive UI
st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="🌍", layout="wide")

# 🏖️ Header
st.title("🌿 Wellness Tourism Package Prediction")
st.markdown("✨ *Predict whether a customer is likely to purchase the Wellness Tourism Package before contacting them.*")

# 📋 Customer Details Section
st.header("👤 Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30)
    TypeofContact = st.selectbox("📞 Contact Type", ["Company Invited", "Self Inquiry"])
    CityTier = st.selectbox("🏙️ City Tier", [1, 2, 3])
    Occupation = st.selectbox("💼 Occupation", ["Salaried", "Freelancer", "Other"])
    Gender = st.selectbox("🚻 Gender", ["Male", "Female"])

with col2:
    NumberOfPersonVisiting = st.number_input("👨‍👩‍👧 Persons Visiting", min_value=1, value=1)
    PreferredPropertyStar = st.selectbox("⭐ Property Star", [3, 4, 5])
    MaritalStatus = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced"])
    NumberOfTrips = st.number_input("✈️ Trips per Year", min_value=0, value=1)
    Passport = st.selectbox("🛂 Passport", ["Yes", "No"])

with col3:
    OwnCar = st.selectbox("🚗 Own Car", ["Yes", "No"])
    NumberOfChildrenVisiting = st.number_input("👶 Children Visiting", min_value=0, value=0)
    Designation = st.text_input("🏷️ Designation")
    MonthlyIncome = st.number_input("💰 Monthly Income", min_value=0, value=50000)

# 🤝 Customer Interaction Section
st.header("📊 Customer Interaction Data")
col4, col5 = st.columns(2)

with col4:
    PitchSatisfactionScore = st.slider("😊 Pitch Satisfaction", min_value=1, max_value=5, value=3)
    ProductPitched = st.selectbox("📦 Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"])

with col5:
    NumberOfFollowups = st.number_input("🔄 Follow-ups", min_value=0, value=1)
    DurationOfPitch = st.number_input("⏱️ Pitch Duration (minutes)", min_value=1, value=10)

# 🛠️ Data Preparation
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': 1 if TypeofContact == "Company Invited" else 0,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': 1 if Gender == "Male" else 0,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])

# 🎯 Prediction
classification_threshold = 0.5
if st.button("🔮 Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "🌟 Likely to Purchase" if prediction == 1 else "❌ Unlikely to Purchase"
    st.success(f"📌 Based on the information provided, the customer is **{result}** the Wellness Tourism Package.")
