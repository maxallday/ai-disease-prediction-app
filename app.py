import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
try:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"Failed to configure Gemini: {str(e)}")
    gemini_model = None

## Setup
st.set_page_config(
    page_title="HealthGuard AI - Multi-Disease Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# Optimized dark theme styling
st.markdown("""
<style>
    .main {
        background-color: #121212;
        color: #e0e0e0;
    }
    .chat-container {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 20px;
        max-height: 500px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #2a3f5f;
        color: #ffffff;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 8px 0;
        max-width: 75%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .ai-message {
        background-color: #2c2c2e;
        color: #e0e0e0;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 8px 0;
        max-width: 75%;
        font-size: 0.95rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .header-text {
        font-size: 2.7rem !important;
        font-weight: bold;
        color: #00acc1 !important;
        text-align: center;
        margin-bottom: 10px;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Title section
st.markdown('<h1 class="header-text">HealthGuard AI</h1>', unsafe_allow_html=True)
st.markdown("##### AI-powered early detection for diabetes and heart disease")
st.markdown("---")

## Sidebar Navigation
with st.sidebar:
    st.markdown("### HealthGuard AI")
    st.markdown("---")
    
    page = st.radio(
        "Navigation", 
        options=["Predictor", "Health Assistant", "About"],
        key="nav_radio"
    )
    
    st.markdown("---")
    st.markdown("### Your Health Stats")
    if "prediction_history" in st.session_state and st.session_state.prediction_history:
        last_pred = st.session_state.prediction_history[-1]
        st.metric(
            f"Last {last_pred['Condition']} Prediction", 
            f"{last_pred['Prediction']} Risk",
            f"{last_pred['Confidence (%)']}% confidence"
        )

#@# Health Assistant Page
if page == "Health Assistant":
    st.header("Health Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello, I'm your HealthGuard AI assistant. How can I help you with your health concerns today?"}
        ]
    
    # Display chat messages
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            if message["role"] == "assistant":
                st.markdown(f'<div class="ai-message">AI: {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="user-message">You: {message["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask health-related questions...", key="chat_input"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.container():
            st.markdown(f'<div class="user-message">You: {prompt}</div>', unsafe_allow_html=True)
        
        # Generate ai response
        if gemini_model:
            with st.spinner("Thinking..."):
                try:
                    # Get context from prediction history if available
                    context = ""
                    if "prediction_history" in st.session_state and st.session_state.prediction_history:
                        context = f"The user has previously checked for {st.session_state.prediction_history[-1]['Condition']} with {st.session_state.prediction_history[-1]['Prediction']} risk."
                    
                    # Generate response using gemini
                    response = gemini_model.generate_content(
                        f"""You are HealthGuard AI, a professional medical assistant. 
                        {context}
                        The user asked: {prompt}
                        Provide a helpful, accurate response in simple terms. 
                        Always recommend consulting a doctor for professional advice."""
                    )
                    ai_response = response.text
                    
                    # Add ai response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                    # Display ai response
                    with st.container():
                        st.markdown(f'<div class="ai-message">AI: {ai_response}</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Sorry, I encountered an error: {str(e)}")
        else:
            st.error("AI service is currently unavailable. Please try again later.")
    
    # Add clear chat button
    if st.button("Clear Conversation", key="clear_chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello, I'm your HealthGuard AI assistant. How can I help you with your health concerns today?"}
        ]
        st.rerun()

## Disease Prediction Section
elif page == "Predictor":
    st.header("Disease Risk Prediction")
    
    # Model paths
    model_paths = {
        "Diabetes": "models/diabetes_model.pkl",
        "Heart Disease": "models/heart_disease_model.pkl"
    }

    # Initialize prediction history
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # Disease selection
    disease = st.selectbox("Select the disease to predict:", ["Diabetes", "Heart Disease"], key="disease_select")

    # Load model
    model_path = model_paths[disease]
    if not os.path.exists(model_path):
        st.error(f"Model for {disease} not found at {model_path}.")
        st.stop()

    model = joblib.load(model_path)

    # Input form
    st.subheader(f"Enter Health Data for {disease} Prediction")

    if disease == "Diabetes":
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, key="pregnancies_input")
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=180, value=70, key="bp_input")
            insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=300, value=80, key="insulin_input")
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, key="pedigree_input")
        with col2:
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=200, value=100, key="glucose_input")
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, key="skin_input")
            bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, key="bmi_input")
            age = st.number_input("Age", min_value=1, max_value=120, value=30, key="age_input")

        user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, diabetes_pedigree, age]])

    elif disease == "Heart Disease":
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 20, 100, 50, key="hd_age_input")
            sex = st.selectbox("Sex", ["Male", "Female"], key="sex_select")
            cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"], key="cp_select")
            bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120, key="hd_bp_input")
            cholesterol = st.number_input("Cholesterol", 100, 400, 200, key="chol_input")
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["Yes", "No"], key="fbs_select")
        with col2:
            rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], key="ecg_select")
            max_hr = st.number_input("Max Heart Rate", 60, 220, 150, key="hr_input")
            ex_angina = st.selectbox("Exercise-Induced Angina?", ["Yes", "No"], key="angina_select")
            oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, key="oldpeak_input")
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], key="slope_select")

        # Encode categorical inputs
        sex_val = 1 if sex == "Male" else 0
        cp_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
        restecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
        st_slope_map = {"Up": 0, "Flat": 1, "Down": 2}
        fbs_val = 1 if fbs == "Yes" else 0
        angina_val = 1 if ex_angina == "Yes" else 0

        user_input = np.array([[
            age,
            sex_val,
            cp_map[cp],
            bp,
            cholesterol,
            fbs_val,
            restecg_map[rest_ecg],
            max_hr,
            angina_val,
            oldpeak,
            st_slope_map[st_slope]
        ]])

    # Prediction
    if st.button("Predict Risk", key="predict_btn"):
        prediction = model.predict(user_input)[0]
        confidence = model.predict_proba(user_input)[0][prediction] * 100 if hasattr(model, "predict_proba") else 100

        result = f"High Risk of {disease}" if prediction == 1 else f"Low Risk of {disease}"

        # Display result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"{result} ({confidence:.2f}% confidence)")
            
            tab1, tab2, tab3 = st.tabs(["Immediate Actions", "Lifestyle Changes", "Medical Advice"])
            
            with tab1:
                st.markdown("What to do right now:")
                if disease == "Diabetes":
                    st.markdown("- Take a 15-minute walk after meals")
                    st.markdown("- Avoid sugary drinks and snacks")
                    st.markdown("- Drink water")
                else:
                    st.markdown("- Practice deep breathing")
                    st.markdown("- Avoid strenuous activity")
                    st.markdown("- Contact your doctor if you have chest pain")
            
            with tab2:
                st.markdown("Long-term improvements:")
                if "user_profile" in st.session_state:
                    if st.session_state.user_profile.get("bmi", 0) > 25:
                        st.markdown(f"- Aim to lose {st.session_state.user_profile['weight'] - (24.9 * (st.session_state.user_profile['height']/100)**2):.1f} kg to reach healthy BMI")
            
            with tab3:
                st.markdown("When to see a doctor:")
                st.markdown("- Schedule a checkup within 2 weeks")
                st.markdown("- Bring these results to your appointment")
                
            if disease == "Heart Disease":
                with st.expander("Emergency Protocol", expanded=True):
                    st.warning("If experiencing these symptoms, call emergency services:")
                    st.markdown("- Chest pain or discomfort")
                    st.markdown("- Shortness of breath")
                    st.markdown("- Lightheadedness")
                    st.markdown("- Pain in jaw/neck/back")
        else:
            st.success(f"{result} ({confidence:.2f}% confidence)")
            
            st.markdown("Keep up the healthy lifestyle:")
            st.markdown("- Eat balanced meals")
            st.markdown("- Exercise regularly")
            st.markdown("- Get adequate sleep")

        # Gauge chart
        st.markdown("Model Confidence")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Prediction Confidence (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightcoral"},
                    {'range': [50, 80], 'color': "gold"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ]
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # Save to history
        st.session_state.prediction_history.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Condition": disease,
            "Prediction": "High" if prediction else "Low",
            "Confidence (%)": round(confidence, 2)
        })

    # Show history
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("Your Health Trends")
        
        df_hist = pd.DataFrame(st.session_state.prediction_history)
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        
        fig = px.line(df_hist, x='Date', y='Confidence (%)', 
                     color='Condition', markers=True,
                     title="Risk Confidence Over Time")
        st.plotly_chart(fig)
        
        # Download
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button("Download History as CSV", csv, "prediction_history.csv", mime='text/csv', key="download_btn")

## About Section 
elif page == "About":
    st.header("About HealthGuard AI")
    
    st.markdown("""
    HealthGuard AI is an intelligent health risk assessment tool designed to provide early warnings for 
    potential health issues using machine learning models.
    """)
    
    st.markdown("### How It Works")
    st.markdown("""
    1. Select a condition you want to assess
    2. Enter your health metrics
    3. Get your risk assessment with confidence percentage
    4. Review personalized recommendations
    """)
    
    st.markdown("### Our Technology")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", "73%", "Diabetes prediction")
    col2.metric("Model Accuracy", "88%", "Heart disease prediction")
    col3.metric("Data Points", "1,000+", "Training samples")
    
    st.markdown("### Disclaimer")
    st.warning("""
    This tool is for educational and informational purposes only. It is not a substitute for professional 
    medical advice, diagnosis, or treatment.
    """)

# Footer
st.markdown("---")
st.markdown("HealthGuard AI | © 2025 | All rights reserved")