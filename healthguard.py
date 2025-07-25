# 📦 Import libraries
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import whisper
import requests
import pandas as pd
from datetime import datetime
import os
import pickle
import logging
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# 🔐 Load API key from environment variable
load_dotenv()
API_KEY = os.getenv("RAPID_API_KEY")

# 🌐 API base and headers
API_BASE = "https://ai-medical-diagnosis-api-symptoms-to-results.p.rapidapi.com/analyzeSymptomsAndDiagnose"
HEADERS = {
    "Content-Type": "application/json",
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "ai-medical-diagnosis-api-symptoms-to-results.p.rapidapi.com"
}

# 🧭 Detect internet connectivity
def is_online():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

ONLINE = is_online()
if not ONLINE:
    st.warning("📴 Offline mode activated. Using local models and advice.")

# 📂 Load offline files
try:
    with open("model_symptoms_v2.pkl", "rb") as f:
        offline_model = pickle.load(f)
    with open("tfidf_v2.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("offline_data.pkl", "rb") as f:
        offline_tips = pickle.load(f)
except Exception as e:
    logging.error("❌ Failed to load offline files: %s", e)
    offline_model, tfidf, offline_tips = None, None, {}

# 🎙️ Load Whisper model
if "model" not in st.session_state:
    st.session_state.model = whisper.load_model("base")

# ⚙️ UI setup
st.set_page_config(page_title="HealthGuard Ultra", page_icon="🩺", layout="centered")
st.title("👨🏾‍⚕️ HealthGuard Ultra")
st.caption("Smart, voice-enabled health companion for real-time analysis and recommendations.")

# 🧠 State init
st.session_state.setdefault("chat", [])
st.session_state.setdefault("last_text", "")
st.session_state.setdefault("prediction_log", [])

# 🔬 Health engine (online + offline)
def healthguard_engine(symptoms, gender="male", year_of_birth=1990):
    if ONLINE:
        payload = {
            "symptoms": [s.strip() for s in symptoms.lower().split(",")],
            "patientInfo": {
                "age": 2025 - int(year_of_birth),
                "gender": gender,
                "height": 165,
                "weight": 65,
                "medicalHistory": ["hypertension"],
                "currentMedications": ["ibuprofen"],
                "allergies": [],
                "lifestyle": {
                    "smoking": False,
                    "alcohol": "rare",
                    "exercise": "light",
                    "diet": "balanced"
                }
            },
            "lang": "en"
        }

        try:
            response = requests.post(API_BASE, headers=HEADERS, json=payload, params={"noqueue": "1"})
            if response.status_code == 200:
                result = response.json().get("result", {})
                analysis = result.get("analysis", {})
                conditions = analysis.get("possibleConditions", [])
                advice = analysis.get("generalAdvice", {})
                resources = result.get("educationalResources", {})

                condition = conditions[0]["condition"] if conditions else "❓ Unknown"
                medications = advice.get("recommendedActions", ["None suggested"])
                tips = advice.get("lifestyleConsiderations", ["No tips available"])
                learn_more = resources.get("preventiveMeasures", ["📚 No information available."])

                return {
                    "condition": condition,
                    "medications": medications,
                    "tips": tips,
                    "info": "\n".join(learn_more)
                }
            else:
                logging.error("⚠️ API failed with code %s", response.status_code)
        except Exception as e:
            logging.error("❌ API error: %s", str(e))

    # 🧠 Offline fallback
    if offline_model and tfidf:
        try:
            X = tfidf.transform([symptoms])
            #prediction = offline_model.predict(X)[0]
            prediction = offline_model.get(symptoms.lower(), "Unknown")

            advice = offline_tips.get(prediction.lower(), "No offline advice available.")
            return {
                "condition": prediction,
                "medications": ["Local advice only"],
                "tips": [advice],
                "info": "Offline mode: results based on local model."
            }
        except Exception as e:
            logging.error("❌ Offline prediction error: %s", str(e))

    # 🚧 Final fallback
    st.error("🚧 Service temporarily unavailable. Please check your connection or refresh.")
    return {
        "condition": "❌ Diagnosis unavailable",
        "medications": [],
        "tips": [],
        "info": (
            "We're currently unable to fetch results due to a connection issue. "
            "Please try again shortly. If the issue persists, contact support."
        )
    }

# 💬 Response generator
def generate_response(msg):
    msg = msg.lower()
    res = healthguard_engine(msg)
    return (
        f"🧠 Likely diagnosis: **{res['condition']}**\n\n"
        f"💊 Medications: {', '.join(res['medications']) or 'None suggested'}\n"
        f"🧘 Lifestyle Tips: {', '.join(res['tips']) or 'No tips available'}\n"
        f"📚 Learn more: {res['info']}"
    )

# 🗨️ Chat display
st.subheader("💬 Chat With Dr. Guard")
for sender, msg in st.session_state.chat[-20:]:
    alignment = "right" if sender.startswith("You") else "left"
    color = "#1a73e8" if alignment == "right" else "#34a853"
    st.markdown(f"<div style='text-align:{alignment}; color:{color}'><strong>{sender}:</strong> {msg}</div>", unsafe_allow_html=True)

# 📥 Text input
text_input = st.text_input("Type your question", placeholder="Try: I have a sore throat and headache...")
if text_input and text_input != st.session_state.last_text:
    st.session_state.chat.append(("You", text_input))
    reply = generate_response(text_input)
    st.session_state.chat.append(("Dr. Guard", reply))
    st.session_state.last_text = text_input

# 🎤 Audio input
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio = b""
    def recv(self, frame):
        self.audio += frame.to_ndarray().tobytes()
        return frame

st.subheader("🎙️ Voice Input")
if st.button("🎤 Start Voice"):
    ctx = webrtc_streamer(
        key="voice", mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True
    )
    if ctx.audio_processor and st.button("📝 Transcribe & Send"):
        audio = ctx.audio_processor.audio
        result = st.session_state.model.transcribe(audio)
        voice_text = result.get("text", "")
        st.session_state.chat.append(("You (voice)", voice_text))
        st.session_state.chat.append(("Dr. Guard", generate_response(voice_text)))
        ctx.audio_processor.audio = b""

# 🩺 Symptom checker
with st.expander("🩺 Symptom Checker"):
    symptoms = st.text_input("Describe your symptoms", value="fever, cough")
    gender = st.selectbox("Gender", ["male", "female"])
    birth_year = st.number_input("Year of Birth", min_value=1930, max_value=2025, value=1990)
    if st.button("📋 Analyze & Recommend"):
        result = healthguard_engine(symptoms, gender, birth_year)
        st.success(f"🧠 Likely Diagnosis: {result['condition']}")
        st.markdown(f"**💊 Suggested Medications:** {', '.join(result['medications'])}")
        st.markdown(f"**🧘 Tips:** {', '.join(result['tips'])}")
        st.markdown(f"**📚 Info:** {result['info']}")


# 📈 Health risk simulator (continued)
with st.expander("📈 Health Risk Simulator"):
    condition = st.selectbox("Condition", ["Diabetes", "Heart Disease"])
    metric1 = st.text_input("Metric 1", placeholder="e.g., glucose level")
    metric2 = st.text_input("Metric 2", placeholder="e.g., blood pressure value")
    
    if st.button("📊 Simulate Risk"):
        try:
            # Mock simulation logic — you can expand this using medical formulas or ML
            risk_score = 0
            if condition == "Diabetes":
                risk_score = float(metric1 or 0) / 200 * 100
            elif condition == "Heart Disease":
                risk_score = float(metric2 or 0) / 180 * 100

            st.markdown(f"🧮 Estimated Risk Score: **{risk_score:.1f}%**")
            if risk_score > 75:
                st.warning("⚠️ High risk. Please consult a healthcare provider.")
            elif risk_score > 40:
                st.info("📌 Moderate risk. Consider a check-up soon.")
            else:
                st.success("✅ Low risk. Keep up the healthy lifestyle!")
        except Exception as e:
            st.error(f"❌ Error in simulation: {e}")
# 🧹 Sidebar log
with st.sidebar.expander("📜 Chat Log", expanded=True):
    for sender, msg in st.session_state.chat:
        icon = "🧍‍♂️" if sender.startswith("You") else "👨🏾‍⚕️"
        st.markdown(f"**{icon} {sender}:** {msg}")
    if st.button("🧹 Clear History"):
        st.session_state.chat.clear()
        st.session_state.last_text = ""

# 📧 Sidebar support link
#support_email = "support@yourdomain.com"
#subject = "HealthGuard Ultra Support"
#mailto_link = f"[📧 Contact Support](mailto:{support_email}?subject={subject.replace(' ', '%20')})"
#st.sidebar.markdown("---")
#st.sidebar.markdown("Need help?")
#st.sidebar.markdown(mailto_link)


with st.expander("📩 Send Feedback or Ask a Question"):
    name = st.text_input("Your Name")
    user_email = st.text_input("Your Email")
    message = st.text_area("Your Message")

    if st.button("📤 Submit Message"):
        if name and user_email and message:
            mailto = f"mailto:{support_email}?subject=Feedback from {name}&body={message}"
            st.markdown(f"[Click here to email us ➡️]({mailto})")
            st.success("✅ Message ready to send via your email client.")
        else:
            st.warning("⚠️ Please fill in all fields to submit.")
# Footer
# 🖤 Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: gray;'>
        © 2025 DevSquad <strong><a href="https://github.com/maxallday" target="_blank">Maxallday</a></strong>
        &nbsp;|&nbsp; 
        <a href="https://github.com/derrickngari" target="_blank">Derrick Ngari</a> 
        &nbsp;|&nbsp; 
        <a href="https://github.com/Bossy-V-Osinde" target="_blank">V.Osinde</a><br>
        <em>“Ensemble, tout est réalisable!!”</em>
    </div>
    
    <div style='text-align: center; font-size: 12px; color: darkgray; margin-top: 10px;'>
        <strong>Disclaimer:</strong> HealthGuard Ultra is an educational tool and does not provide medical diagnoses or treatment.
        Always consult a licensed healthcare provider for any concerns regarding your health.
    </div>
    """,
    unsafe_allow_html=True
)
