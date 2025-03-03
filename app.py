import streamlit as st
import spacy
import torch
import subprocess
from transformers import pipeline
from gtts import gTTS
import os

# ✅ Ensure Dependencies are Compatible
subprocess.run(["pip", "install", "--upgrade", "pip"])
subprocess.run(["pip", "install", "numpy<2"])

# ✅ Ensure Spacy Model is Installed
def install_models():
    try:
        spacy.load("en_core_web_sm")  # Check if model exists
    except OSError:
        st.warning("Downloading Spacy model: en_core_web_sm...")
        result = subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Failed to install Spacy model: {result.stderr}")
            st.stop()

install_models()  # Run this before loading models

# ✅ Cache model loading to improve performance
@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception as e:
        st.error(f"Failed to load Spacy model: {e}")
        return None, None
    
    device = 0 if torch.cuda.is_available() else -1
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    except Exception as e:
        st.error(f"Failed to load transformers model: {e}")
        return None, None

    return nlp, summarizer

nlp, summarizer = load_models()

if nlp is None or summarizer is None:
    st.error("Model loading failed. Please check logs.")
    st.stop()

# ✅ Initialize Session State
if "summary" not in st.session_state:
    st.session_state.summary = None

# ✅ UI: Streamlit App
st.title("AI-Based Text Summarizer & Speech Converter")
st.write("Upload a text file to summarize and listen.")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8").strip()

    if len(text) < 10:
        st.error("File content is too short to summarize.")
    else:
        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                try:
                    st.session_state.summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                    st.subheader("Summary")
                    st.write(st.session_state.summary)
                    
                    # ✅ Download Summary
                    st.download_button("Download Summary", st.session_state.summary, file_name="summary.txt", mime="text/plain")
                except Exception as e:
                    st.error(f"Summarization error: {e}")

# ✅ Convert to Speech (Now Works Properly)
if st.button("Convert to Speech"):
    if st.session_state.summary:  # ✅ Now it persists across interactions
        with st.spinner("Generating Audio..."):
            try:
                audio_path = "summary_audio.mp3"
                gTTS(st.session_state.summary, lang="en").save(audio_path)
                st.audio(audio_path, format="audio/mp3")
            except Exception as e:
                st.error(f"Text-to-speech error: {e}")
    else:
        st.warning("Please generate a summary first before converting to speech.")
