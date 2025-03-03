import streamlit as st
import spacy
import re
import torch
from transformers import pipeline
from gtts import gTTS
import os

# ✅ FIXED: Load models without caching (Prevents UnboundLocalError)
def load_models():
    try:
        spacy.cli.download("en_core_web_sm")  # Ensure model is downloaded
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        return None, None  # Prevents UnboundLocalError

    # ✅ FIXED: Ensure Summarizer loads correctly
    try:
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    except Exception as e:
        st.error(f"Error loading summarizer: {e}")
        return nlp, None  # Prevents crash

    return nlp, summarizer

nlp, summarizer = load_models()

# ✅ FIXED: Ensure models are loaded before proceeding
if nlp is None or summarizer is None:
    st.error("Model loading failed. Try restarting the app.")
    st.stop()  # Stop execution if models fail to load

# Function to clean text efficiently
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9.,!? ]+', '', text).strip()

# Efficient sentence segmentation
def segment_text(text):
    return " ".join([sent.text for sent in nlp(text).sents])

# Generate summary only when needed (Cached)
@st.cache_data
def generate_summary(text, max_length=150, min_length=50):
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

# Convert text to speech
def text_to_speech(text):
    audio_path = "summary_audio.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(audio_path)
    return audio_path

# Streamlit UI
st.set_page_config(page_title="AI-Based Notes Reader", layout="centered")

st.markdown("<h1 style='text-align: center;'>AI-Based Notes Reader</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Upload your text file to generate a summary and listen to it.</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
summary = None

if uploaded_file is not None:
    if uploaded_file.size > 500000:
        st.error("File too large! Please upload a smaller file.")
    else:
        text = uploaded_file.read().decode("utf-8")
        text = clean_text(text)
        text = segment_text(text)

        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = generate_summary(text)
                st.subheader("AI-Generated Summary:")
                st.write(summary)

                st.download_button(label="Download Summary", data=summary, file_name="summary.txt", mime="text/plain")

if summary and st.button("Convert to Speech"):
    with st.spinner("Generating Audio..."):
        audio_file = text_to_speech(summary)
        st.audio(audio_file, format="audio/mp3")
