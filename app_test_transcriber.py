import streamlit as st
import speech_recognition as sr

recognizer = sr.Recognizer()
transcription_text = ""

def transcribe_speech():
    global transcription_text
    transcription_text = ""
    with sr.Microphone() as source:
        st.session_state["status"] = "Recording"
        st.write("Listening continuously... (say 'stop' to end)")

        while st.session_state["listening"]:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source)

            try:
                text = recognizer.recognize_google(audio_data).lower()
                transcription_text += text + " "
                st.session_state["transcription"] = transcription_text

                if "stop" in text:
                    st.session_state["listening"] = False
                    st.session_state["status"] = "Stopped"
            except sr.UnknownValueError:
                st.session_state["transcription"] = "Could not understand the audio."
            except sr.RequestError:
                st.session_state["transcription"] = "API unavailable."

# Streamlit app structure
st.title("Real-time Speech Recognition")

# Initialize session state for status and transcription
if "status" not in st.session_state:
    st.session_state["status"] = "Stopped"
if "transcription" not in st.session_state:
    st.session_state["transcription"] = ""
if "listening" not in st.session_state:
    st.session_state["listening"] = False

# Start and stop button with animation
col1, col2 = st.columns(2)

with col1:
    if st.button("Start Listening") and not st.session_state["listening"]:
        st.session_state["listening"] = True
        transcribe_speech()

with col2:
    if st.button("Stop Listening"):
        st.session_state["listening"] = False
        st.session_state["status"] = "Stopped"

if st.session_state["status"] == "Recording":
    st.markdown("🎙️ **Recording...**")
else:
    st.markdown("⏹️ **Stopped**")

st.markdown("### Transcription")
st.write(st.session_state["transcription"])
