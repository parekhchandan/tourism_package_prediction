import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="Chandan2312/tourism_package_prediction",
    filename="best_customer_model_v1.joblib"
)

# Load the model
model = joblib.load(model_path)

# 🎉 Attractive UI
st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="🌍", layout="centered")

# 🏖️ Header
st.title("🌿 Wellness Tourism Package Prediction")
st.markdown("✨ *Predict whether a customer is likely to purchase the Wellness Tourism Package before contacting them.*")

# 📋 Customer Details Section
st.header("👤 Customer Details")
Age = st.number_input("🎂 Age of Customer", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("📞 Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("🏙️ City Tier", [1, 2, 3])
Occupation = st.selectbox("💼 Occupation", ["Salaried", "Freelancer", "Other"])
Gender = st.selectbox("🚻 Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("👨‍👩‍👧 Number of Persons Visiting", min_value=1, value=1)
PreferredPropertyStar = st.selectbox("⭐ Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("✈️ Average Trips per Year", min_value=0, value=1)
Passport = st.selectbox("🛂 Passport", ["Yes", "No"])
OwnCar = st.selectbox("🚗 Own Car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("👶 Number of Children Visiting", min_value=0, value=0)
Designation = st.text_input("🏷️ Designation")
MonthlyIncome = st.number_input("💰 Monthly Income", min_value=0, value=50000)

# 🤝 Customer Interaction Section
st.header("📊 Customer Interaction Data")
PitchSatisfactionScore = st.slider("😊 Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
ProductPitched = st.selectbox("📦 Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"])
NumberOfFollowups = st.number_input("🔄 Number of Follow-ups", min_value=0, value=1)
DurationOfPitch = st.number_input("⏱️ Duration of Pitch (minutes)", min_value=1, value=10)

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
