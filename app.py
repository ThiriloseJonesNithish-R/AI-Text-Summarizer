import streamlit as st
import spacy
import torch
from transformers import pipeline
from gtts import gTTS
import os
import subprocess

# ✅ Ensure Spacy Model is Installed Before Loading
def install_models():
    try:
        spacy.load("en_core_web_sm")  # Try loading the model
    except OSError:
        st.warning("Downloading Spacy model: en_core_web_sm...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)

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

# ✅ UI: Streamlit App
st.title("AI-Based Notes Reader")
st.write("Upload a text file to summarize and listen.")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8").strip()
    
    if len(text) < 10:
        st.error("File content is too short to summarize.")
    else:
        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                st.subheader("Summary")
                st.write(summary)
                
                # ✅ Download Summary
                st.download_button("Download Summary", summary, file_name="summary.txt", mime="text/plain")

        if st.button("Convert to Speech"):
            with st.spinner("Generating Audio..."):
                audio_path = "summary_audio.mp3"
                gTTS(summary, lang="en").save(audio_path)
                st.audio(audio_path, format="audio/mp3")
