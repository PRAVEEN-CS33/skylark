import streamlit as st
import time
from separator import separate_vocals
from tempfile import NamedTemporaryFile
from pathlib import Path
from profanity_filter import profanity_filter_add
from noise_remover import noise_removal_filter
from pydub import AudioSegment
from io import BytesIO
import torch
import numpy as np
import whisper
import speech_recognition as sr
from deep_translator import GoogleTranslator
# from TTS.api import TTS
# tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)

recognizer = sr.Recognizer()
whisper_model = whisper.load_model("small")
device="cuda:0" if torch.cuda.is_available() else "cpu"
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

st.set_page_config(page_title="Skylark", layout="wide")

st.markdown("<h1 style='text-align: center;'>Welcome to Skylark</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Navigation")
    option = st.radio(
        "Select an option:",
        ("Synthesis", "LiveAudio", "Stream Translation", "Audio Summary", "AudioToText", "Vocal Separation", "Remove Noise", "TextToAudio")
    )

if "audio_file" not in st.session_state:
    st.session_state.audio_file = None

st.subheader("Upload an Audio File")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.session_state.audio_file = audio_file
    suffix = Path(audio_file.name).suffix[1:]

if st.session_state.audio_file is not None:
    st.audio(st.session_state.audio_file, format="audio/wav")
else:
    st.info("Please upload an audio file to use in various options.")

if option == "Synthesis":
    st.subheader("Audio Synthesis")
    st.write("Configure your synthesis options below:")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        profanity_filter = st.checkbox("Enable Profanity Filter",value=True)
    with col2:
        noise_removal = st.checkbox("Enable Advanced Noise Removal",value=True)
    with col3:
        vocal_separation = st.checkbox("Enable Vocal Separation",value=True)

    with col2:
        if st.button("Synthesize"):
            tmp_path = "C:/Users/samue/Desktop/naval_hacko/Speech Recognition and Splitter/final_skylark/2_audio.wav"    
            tmp_path = "C:/Users/samue/Desktop/naval_hacko/Speech Recognition and Splitter/final_skylark/final_sample.wav"         
            split_audios = separate_vocals(tmp_path)
            detected_voices = 3
            for ind in range(detected_voices):
                curr_audio = split_audios[:, :, ind]
                # if profanity_filter:
                #    curr_audio, transcription = profanity_filter_add(curr_audio, 8000)
                if noise_removal:
                    curr_audio = noise_removal_filter(curr_audio)
                
                #st.write(transcription)
                st.write(f"Detected Voice {ind+1}")
                st.audio(curr_audio.detach().cpu().numpy(), sample_rate=8000, format="audio/wav")
                # transcription = transcribe_audio(st.session_state.tmp_file_name)
                # st.write("Transcription Result:")
                # st.write(transcription)
                # st.write("- Profanity Filter: Enabled")
                   
elif option == "LiveAudio":
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
        st.markdown("")

    st.markdown("### Transcription")
    st.write(st.session_state["transcription"])

elif option == "Stream Translation":
    st.subheader("Stream Translation")
    if st.button("Translate"):
        tmp_path = "C:/Users/samue/Desktop/naval_hacko/Speech Recognition and Splitter/final_skylark/LOTR.wav"    
        transcription = whisper_model.transcribe(tmp_path, word_timestamps=True, language=None)
        translated_text = GoogleTranslator(source="auto", target="hi").translate(transcription["text"])
        st.write("Transcribed text")
        st.write(translated_text)
        st.write("Translated text")
        st.audio("reSpoken.wav")

elif option == "Audio Summary":
    st.subheader("Audio Summary")
    st.write("Functionality coming soon...")

elif option == "AudioToText":
    st.subheader("Audio to Text Conversion")
    if st.session_state.audio_file:
        transcription = transcribe_audio(st.session_state.audio_file)
        st.write("Transcription Result:")
        st.write(transcription)
    else:
        st.info("Please upload an audio file to convert.")

elif option == "Vocal Separation":
    st.subheader("Vocal Separation")
    st.write("Functionality coming soon...")

elif option == "Remove Noise":
    st.subheader("Noise Removal")
    st.write("Functionality coming soon...")

elif option == "TextToAudio":
    st.subheader("Text to Audio Conversion")
    st.write("Functionality coming soon...")