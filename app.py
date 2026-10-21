import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Patient Churn Analytics Dashboard", 
    page_icon="🏥", 
    layout="wide"
)

# VIP Professional CSS Styling & Background Colors
st.markdown("""
    <style>
    /* Global background and font styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }
    .main {
        background-color: transparent;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Custom Styling for Predict Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        height: 55px;
        border: none;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 6px 15px rgba(37, 99, 235, 0.4);
    }
    /* Creator Badge Styling */
    .creator-badge {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #38bdf8;
        padding: 10px 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        letter-spacing: 1px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Model, Scaler aur Columns load karna
@st.cache_resource
def load_files():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    cols = joblib.load('train_columns.pkl')
    return model, scaler, cols

model_streamlit, scaler, train_columns = load_files()

# Sidebar Setup with VIP Branding
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital-3.png", width=70)
    st.subheader("VIP Control Panel")
    st.info("Enterprise-grade patient retention and churn intelligence dashboard.")
    st.markdown("---")
    st.markdown("### 👨‍💻 Developed By:")
    st.markdown("<div class='creator-badge'>MUHAMMAD AHSAN</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("**Status:** System Online 🟢")

# Header Section
st.title("🏥 Executive Patient Retention & Churn Dashboard")
st.markdown("<p style='color: #475569; font-size: 16px; font-weight: 500;'>Advanced predictive analytics platform to monitor patient engagement, assess drop-off risk, and optimize healthcare delivery.</p>", unsafe_allow_html=True)
st.write("---")

st.subheader("📋 Patient Clinical & Behavioral Profile:")

# 2. User Inputs in 3 clean VIP columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Demographic Info")
    age = st.slider("Patient Age", 18, 100, 45)
    gender = st.selectbox("Gender", ["Female", "Male"])
    state = st.selectbox("State", ["CA", "FL", "GA", "IL", "MI", "NC", "NY", "OH", "PA", "TX", "Other"])
    distance = st.number_input("Distance To Facility Miles", min_value=0.0, max_value=100.0, value=5.0)

with col2:
    st.markdown("#### 📅 Activity & Visits")
    tenure = st.number_input("Tenure Months", min_value=0, max_value=120, value=12)
    visits = st.number_input("Visits Last Year", min_value=0, max_value=50, value=5)
    missed_app = st.number_input("Missed Appointments", min_value=0, max_value=20, value=1)
    days_since_visit = st.number_input("Days Since Last Visit", min_value=0, max_value=365, value=30)
    referrals_made = st.number_input("Referrals Made", min_value=0, max_value=20, value=0)

with col3:
    st.markdown("#### ⭐ Experience & Services")
    satisfaction = st.slider("Overall Satisfaction (1-5)", 1, 5, 3)
    wait_time = st.slider("Wait Time Satisfaction (1-5)", 1, 5, 3)
    staff_satisfaction = st.slider("Staff Satisfaction (1-5)", 1, 5, 3)
    provider_rating = st.slider("Provider Rating (1-5)", 1, 5, 4)
    cost = st.number_input("Avg Out Of Pocket Cost", min_value=0.0, max_value=1000.0, value=50.0)
    billing_issues = st.selectbox("Billing Issues", [0, 1])
    portal_usage = st.selectbox("Portal Usage", [0, 1])
    specialty = st.selectbox("Specialty", ["Cardiology", "Family Medicine", "General Practice", "Internal Medicine", "Neurology", "Orthopedics", "Pediatrics", "Other"])
    insurance = st.selectbox("Insurance Type", ["Medicaid", "Medicare", "Private", "Self-Pay"])

st.write("---")

# 3. Predict Button aur Data Processing (Working 100% same)
if st.button("🚀 Run Executive Churn Analysis"):
    try:
        input_df = pd.DataFrame(0, index=[0], columns=train_columns)
        
        input_df['Age'] = age
        input_df['Tenure_Months'] = tenure
        input_df['Visits_Last_Year'] = visits
        input_df['Missed_Appointments'] = missed_app
        input_df['Days_Since_Last_Visit'] = days_since_visit
        input_df['Overall_Satisfaction'] = satisfaction
        input_df['Wait_Time_Satisfaction'] = wait_time
        input_df['Staff_Satisfaction'] = staff_satisfaction
        input_df['Provider_Rating'] = provider_rating
        input_df['Avg_Out_Of_Pocket_Cost'] = cost
        input_df['Billing_Issues'] = billing_issues
        input_df['Portal_Usage'] = portal_usage
        input_df['Referrals_Made'] = referrals_made
        input_df['Distance_To_Facility_Miles'] = distance
        
        g_col = f"Gender_{gender}"
        if g_col in input_df.columns:
            input_df[g_col] = 1
            
        s_col = f"State_{state}"
        if s_col in input_df.columns:
            input_df[s_col] = 1
            
        sp_col = f"Specialty_{specialty}"
        if sp_col in input_df.columns:
            input_df[sp_col] = 1
            
        ins_col = f"Insurance_Type_{insurance}"
        if ins_col in input_df.columns:
            input_df[ins_col] = 1
            
        input_scaled = scaler.transform(input_df)
        prediction = model_streamlit.predict(input_scaled)
        
        # Calculate probability score
        if hasattr(model_streamlit, "predict_proba"):
            proba = model_streamlit.predict_proba(input_scaled)
            churn_prob = proba[0][1] * 100
        else:
            churn_prob = 100.0 if prediction[0] == 1 else 0.0

        st.write("### 📊 Executive Intelligence Report:")
        res1, res2 = st.columns(2)
        
        with res1:
            if prediction[0] == 1:
                st.error("🚨 High Churn Risk Detected: Immediate Patient Retention Intervention Required!")
            else:
                st.success("✨ Low Churn Risk: Patient Expected to Maintain Active Status.")
                
        with res2:
            st.metric(label="Calculated Churn Probability Score", value=f"{churn_prob:.1f}%")
            st.progress(int(churn_prob))
            
    except Exception as e:
        st.error(f"Analysis error encountered: {e}")
