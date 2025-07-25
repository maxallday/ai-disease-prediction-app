import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import whisper
import pandas as pd
from datetime import datetime
import tempfile

# 🧠 Load Whisper model
#model = whisper.load_model("base")
if "model" not in st.session_state:
    st.session_state.model = whisper.load_model("base")



# ⚙️ App Config
st.set_page_config(page_title="HealthGuard AI", page_icon="🩺", layout="centered")
st.title("👨🏾‍⚕️ HealthGuard AI")
st.caption("Ask health questions, simulate risks, and speak with your AI doctor.")

# 💾 Session State
st.session_state.setdefault("chat", [])
st.session_state.setdefault("last_text", "")
st.session_state.setdefault("prediction_log", [])

# 👨🏾‍⚕️ Doctor Info
with st.expander("👨🏾‍⚕️ Meet Dr. Guard"):
    st.markdown("""
    **Specializations:**  
    - General Practice  
    - Preventive Health  
    - Risk Simulation  
    - Patient Education  
    """)

# 💬 Response Generator
def generate_response(msg):
    msg = msg.lower()
    if "glucose" in msg:
        return "🩺 Fasting glucose should be 70–99 mg/dL."
    elif "bmi" in msg:
        return "🩺 A healthy BMI is between 18.5 and 24.9."
    elif "blood pressure" in msg or "bp" in msg:
        return "🩺 Normal blood pressure is under 120/80 mmHg."
    elif "symptoms" in msg:
        return "🩺 Describe your symptoms and I’ll guide you."
    else:
        return "🩺 Try asking about glucose, BMI, blood pressure, or health risks."

# 🩺 Risk Prediction Logic
def simulate_prediction(condition, val1, val2):
    try:
        v1, v2 = float(val1), float(val2)
        total = v1 + v2
        thresholds = {"Diabetes": 200, "Heart Disease": 160}
        risk = total > thresholds.get(condition, 180)
        confidence = 88.5 if risk else 93.2
        label = "High" if risk else "Low"
        advice = "- Monitor your values\n- Eat well\n- Stay active"
        return label, confidence, advice
    except:
        return "Error", 0, "⚠️ Please enter valid numbers."

# 💬 WhatsApp-Style Chat Display
st.subheader("💬 Chat")
chat_container = st.container()
with chat_container:
    for sender, msg in st.session_state.chat[-20:]:
        if sender.startswith("You"):
            st.markdown(f"<div style='text-align:right; color:#1a73e8'><strong>You:</strong> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left; color:#34a853'><strong>{sender}:</strong> {msg}</div>", unsafe_allow_html=True)

# 🧍‍♂️ Text Input
text_input = st.text_input("Ask a question:", placeholder="e.g. What's normal BP?")
if text_input and text_input != st.session_state.last_text:
    st.session_state.chat.append(("You", text_input))
    reply = generate_response(text_input)
    st.session_state.chat.append(("Dr. Guard", reply))
    st.session_state.last_text = text_input

# 🎤 Voice Input Trigger
st.subheader("🎙️ Speak Your Question")
voice_active = st.button("🎤 Start Voice Input")
if voice_active:
    class AudioProcessor(AudioProcessorBase):
        def __init__(self): self.audio = b""
        def recv(self, frame):
            self.audio += frame.to_ndarray().tobytes()
            return frame

    webrtc_ctx = webrtc_streamer(
        key="voice-chat",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if webrtc_ctx.audio_processor and st.button("📝 Transcribe & Send Voice"):
        audio = webrtc_ctx.audio_processor.audio
        if audio:
            #voice_text = transcribe_audio(audio)
            voice_text = st.session_state.model.transcribe(audio)

            st.success(f"🗣️ You said: {voice_text}")
            st.session_state.chat.append(("You (voice)", voice_text))
            response = generate_response(voice_text)
            st.session_state.chat.append(("Dr. Guard", response))
            webrtc_ctx.audio_processor.audio = b""

# 🧪 Risk Simulation Panel
st.divider()
with st.expander("⚕️ Run a Risk Simulation"):
    condition = st.selectbox("Choose condition", ["Diabetes", "Heart Disease"])
    metric1 = st.text_input("Input 1 (e.g. glucose, age)")
    metric2 = st.text_input("Input 2 (e.g. BMI, BP)")
    if st.button("📊 Predict Risk"):
        label, confidence, advice = simulate_prediction(condition, metric1, metric2)
        st.success(f"{label} Risk of {condition} ({confidence:.1f}% confidence)")
        st.markdown(advice)
        st.session_state.prediction_log.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Condition": condition,
            "Prediction": label,
            "Confidence (%)": confidence,
            "Inputs": {"Metric 1": metric1, "Metric 2": metric2}
        })

# 📁 Export Prediction History
if st.session_state.prediction_log:
    df = pd.DataFrame(st.session_state.prediction_log)
    st.download_button("📁 Export Predictions", df.to_csv(index=False), "healthguard_predictions.csv", "text/csv")

# 🧹 Clear Chat Option

# 📜 Sidebar Chat + Clear Button
with st.sidebar.expander("📜 Full Chat History", expanded=True):
#   for sender, msg in st.session_state.chat:
 #       icon = "🧍‍♂️" if sender.startswith("You") else "👨🏾‍⚕️"
#        st.markdown(f"**{icon} {sender}:** {msg}")
    
    st.divider()
    if st.button("🧹 Clear Chat History (Sidebar)"):
        st.session_state.chat.clear()
        st.session_state.last_text = ""
