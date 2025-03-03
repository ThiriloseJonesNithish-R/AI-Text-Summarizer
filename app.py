import streamlit as st
import spacy
import re
import torch
from transformers import pipeline
from gtts import gTTS
import os

# ✅ Fix: Load models once and cache to reduce memory usage
@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    # ✅ Fix: Use GPU if available, else CPU
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    
    return nlp, summarizer

nlp, summarizer = load_models()

# ✅ Fix: Clean text efficiently
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9.,!? ]+', '', text).strip()

# ✅ Fix: Sentence segmentation for better summarization
def segment_text(text):
    return " ".join([sent.text for sent in nlp(text).sents])

# ✅ Fix: Cache summary generation to prevent recomputation
@st.cache_data
def generate_summary(text, max_length=150, min_length=50):
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

# ✅ Convert summary to speech
def text_to_speech(text):
    audio_path = "summary_audio.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(audio_path)
    return audio_path

# ✅ Streamlit UI
st.set_page_config(page_title="AI-Based Notes Reader", layout="centered")

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
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>AI-Based Notes Reader</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subtitle'>Upload a text file to generate a summary and listen to it.</h2>", unsafe_allow_html=True)

# ✅ Fix: File Upload with Size Limit
uploaded_file = st.file_uploader("Upload a text file (Max: 300KB)", type=["txt"])
if uploaded_file:
    if uploaded_file.size > 300000:  # 300 KB limit
        st.error("File too large! Please upload a smaller file.")
    else:
        text = uploaded_file.read().decode("utf-8")
        text = clean_text(text)
        text = segment_text(text)

        # ✅ Generate Summary Button
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = generate_summary(text)
                st.subheader("AI-Generated Summary:")
                st.write(summary)
                
                # ✅ Download Summary as Text File
                st.download_button("Download Summary", data=summary, file_name="summary.txt", mime="text/plain")

        # ✅ Convert to Speech Button
        if st.button("Convert to Speech"):
            with st.spinner("Generating Audio..."):
                audio_file = text_to_speech(summary)
                st.audio(audio_file, format="audio/mp3")
