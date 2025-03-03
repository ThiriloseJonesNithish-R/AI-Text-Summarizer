import streamlit as st
import nltk
import spacy
import re
from transformers import pipeline
from gtts import gTTS
import os
os.system("pip install --upgrade torch torchvision torchaudio")

# Download necessary NLP models
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load AI Summarizer Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,!? ]', '', text)
    return text.strip()

# Function to segment sentences
def sentence_segmentation(text):
    doc = nlp(text)
    return " ".join([sent.text for sent in doc.sents])

# Function to generate summary
def generate_summary(text, max_length=150, min_length=50):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Function to convert text to speech
def text_to_speech(text):
    if not os.path.exists("sounds"):
        os.makedirs("sounds")  # Create folder if it doesn't exist
    audio_path = "sounds/summary_audio.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(audio_path)
    return audio_path

# Streamlit UI
st.set_page_config(page_title="AI-Based Notes Reader", layout="centered")

# Apply dark theme with silver UI elements and hover effects
st.markdown(
    """
    <style>
    body {
        background: radial-gradient(circle, #0E1A40, #000000);
        color: white;
    }
    .stApp {
        background: radial-gradient(circle, #0E1A40, #000000);
        color: white;
    }
    .stButton>button {
        background-color: silver !important;
        color: black !important;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: gray !important;
        color: white !important;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 1.5em;
    }
    .download-link {
        text-decoration: none;
        color: #1F6FEB;
        font-weight: bold;
    }
    .download-link:hover, .download-link:focus {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>AI-Based Notes Reader</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subtitle'>Upload your text file to generate a summary and listen to it.</h2>", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

    # Process text
    text = clean_text(text)
    text = sentence_segmentation(text)

    # Generate summary
    summary = generate_summary(text)
    st.subheader("AI-Generated Summary:")
    st.write(summary)

    # Provide Download Option for Summary
    with open("summary.txt", "w") as f:
        f.write(summary)
    st.markdown(f'<a href="summary.txt" download="summary.txt" class="download-link">Download Summary</a>', unsafe_allow_html=True)

    # Convert summary to speech
    if st.button("Convert to Speech"):
        audio_file = text_to_speech(summary)
        st.audio(audio_file, format="audio/mp3")
