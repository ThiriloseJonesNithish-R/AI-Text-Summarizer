import streamlit as st
import spacy
import re
import torch
from transformers import pipeline
from gtts import gTTS
import os

# Load models once (WITHOUT caching to avoid UnboundLocalError)
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Disable unused components
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    # Load summarizer only if GPU/CPU supports it
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    
    return nlp, summarizer

# Call load_models without caching to avoid errors
nlp, summarizer = load_models()

# Function to clean text efficiently
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9.,!? ]+', '', text).strip()

# Efficient sentence segmentation
def segment_text(text):
    return " ".join([sent.text for sent in nlp(text).sents])

# Generate summary only when needed (with caching)
@st.cache_data
def generate_summary(text, max_length=150, min_length=50):
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

# Convert text to speech
def text_to_speech(text):
    audio_path = "summary_audio.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(audio_path)
    return audio_path

# Streamlit UI with Optimized Performance
st.set_page_config(page_title="AI-Based Notes Reader", layout="centered")

st.markdown("<h1 style='text-align: center;'>AI-Based Notes Reader</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Upload your text file to generate a summary and listen to it.</h2>", unsafe_allow_html=True)

# File Upload with Size Limit
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
summary = None  # Initialize summary variable

if uploaded_file is not None:
    if uploaded_file.size > 500000:  # Limit file size to 500 KB
        st.error("File too large! Please upload a smaller file.")
    else:
        text = uploaded_file.read().decode("utf-8")
        text = clean_text(text)
        text = segment_text(text)

        # Summary Generation (Runs only on demand)
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = generate_summary(text)
                st.subheader("AI-Generated Summary:")
                st.write(summary)

                # Download Summary (Served as a downloadable text file)
                st.download_button(label="Download Summary", data=summary, file_name="summary.txt", mime="text/plain")

# Convert summary to speech only after it's generated
if summary and st.button("Convert to Speech"):
    with st.spinner("Generating Audio... "):
        audio_file = text_to_speech(summary)
        st.audio(audio_file, format="audio/mp3")
