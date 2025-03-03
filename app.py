import streamlit as st
import spacy
import torch
import re
import subprocess
from transformers import pipeline
from gtts import gTTS
import os

# ✅ Ensure this is the FIRST Streamlit command
st.set_page_config(page_title="AI-Based Notes Reader", layout="centered")

# Ensure SpaCy model is installed
@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name, disable=["ner"])  # ✅ Enable the parser
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name, disable=["ner"])  # ✅ Enable the parser

nlp = load_spacy_model()


# Load summarization model
@st.cache_resource
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

summarizer = load_summarizer()

# Function to clean text
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9.,!? ]+', '', text).strip()

# Efficient sentence segmentation
def segment_text(text):
    return " ".join([sent.text for sent in nlp(text).sents])

# Generate summary with caching
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
st.markdown(
    """
    <style>
    body, .stApp { background: radial-gradient(circle, #0E1A40, #000000); color: white; }
    .stButton>button {
        background-color: silver !important; color: black !important; border-radius: 8px;
        padding: 10px; font-size: 16px; transition: 0.3s;
    }
    .stButton>button:hover { background-color: gray !important; color: white !important; }
    .title { text-align: center; font-size: 2.5em; font-weight: bold; }
    .subtitle { text-align: center; font-size: 1.5em; }
    .download-link { text-decoration: none; color: #1F6FEB; font-weight: bold; visibility: hidden; }
    .download-link:hover, .download-link:focus { visibility: visible; text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>AI-Based Notes Reader</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subtitle'>Upload your text file to generate a summary and listen to it.</h2>", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
if uploaded_file is not None:
    if uploaded_file.size > 500000:  # Limit file size to 500 KB
        st.error("File too large! Please upload a smaller file.")
    else:
        text = uploaded_file.read().decode("utf-8")
        text = clean_text(text)
        text = segment_text(text)

        # Generate Summary Button
if st.button("Generate Summary"):
    with st.spinner("Summarizing..."):
        summary = generate_summary(text)
        st.session_state["summary"] = summary  # ✅ Store in session state
        st.subheader("AI-Generated Summary:")
        st.write(summary)

# ✅ Display the summary & download button after it's generated
if "summary" in st.session_state and st.session_state["summary"]:
    st.subheader("AI-Generated Summary:")
    st.write(st.session_state["summary"])

    # ✅ Ensure download button appears after summary is generated
    st.download_button(
        label="Download Summary",
        data=st.session_state["summary"],
        file_name="summary.txt",
        mime="text/plain"
    )

# Convert to Speech Button
if st.button("Convert to Speech"):
    if "summary" in st.session_state and st.session_state["summary"]:
        with st.spinner("Generating Audio..."):
            audio_file = text_to_speech(st.session_state["summary"])
            st.audio(audio_file, format="audio/mp3")

        # ✅ Persist summary after conversion
        st.subheader("AI-Generated Summary:")
        st.write(st.session_state["summary"])
    else:
        st.error("Please generate a summary first!")
