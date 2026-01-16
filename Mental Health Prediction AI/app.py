import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import string
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
DATA_PATH = "mental_health_data.csv" # Or "Combined Data.csv" if you renamed it
MODEL_PATH = "mental_health_lstm_model.h5"
MAX_LEN = 100
VOCAB_SIZE = 8000

# Label Mapping (Must match training exactly)
label_mapping = {
    0: 'Anxiety',
    1: 'Normal',
    2: 'Depression',
    3: 'Suicidal',
    4: 'Stress',
    5: 'Bipolar',
    6: 'Personality Disorder'
}

# ==========================================
# 2. LOAD RESOURCES (CACHED)
# ==========================================
# We use @st.cache_resource so it only loads once, not every time you click a button
@st.cache_resource
def load_resources():
    # --- A. Load and Clean Data (for Tokenizer) ---
    try:
        df = pd.read_csv(DATA_PATH)
        df = df[['statement', 'status']]
        df.dropna(inplace=True)
    except FileNotFoundError:
        st.error(f"Error: Could not find {DATA_PATH}. Please make sure it is in the same folder.")
        return None, None

    # Cleaning function (Same as main.py)
    def clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['statement'] = df['statement'].apply(clean_text)

    # --- B. Re-Train Tokenizer ---
    # We re-train it quickly to ensure it maps words to numbers exactly like the training phase
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"], vocab_size=VOCAB_SIZE)
    
    # Train on the text data
    tokenizer.train_from_iterator(df['statement'].tolist(), trainer)

    # --- C. Load the Trained Model ---
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except OSError:
        st.error(f"Error: Could not find {MODEL_PATH}. Did you run main.py to train it first?")
        return None, None
        
    return tokenizer, model

# Load everything
tokenizer, model = load_resources()

# ==========================================
# 3. USER INTERFACE
# ==========================================
st.title("🧠 Mental Health Prediction AI")
st.write("Type a sentence describing how you feel, and the AI will analyze the potential mental health status.")

# Input Area
user_input = st.text_area("How are you feeling?", height=100, placeholder="e.g., I feel very anxious and panic about small things...")

# Prediction Button
if st.button("Analyze Status"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        if tokenizer is not None and model is not None:
            # 1. Clean the user input
            cleaned_input = user_input.lower()
            cleaned_input = re.sub(r'http[s]?://\S+|www\.\S+', '', cleaned_input)
            cleaned_input = cleaned_input.translate(str.maketrans('', '', string.punctuation))
            cleaned_input = re.sub(r'\w*\d\w*', '', cleaned_input)
            cleaned_input = re.sub(r'\s+', ' ', cleaned_input).strip()

            # 2. Tokenize
            encoded = tokenizer.encode(cleaned_input).ids
            
            # 3. Pad (Ensure it is 100 length)
            # We pad with 0s at the end
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                [encoded], maxlen=MAX_LEN, padding='post', truncating='post'
            )

            # 4. Predict
            prediction_probs = model.predict(padded)
            predicted_class_id = np.argmax(prediction_probs, axis=1)[0]
            predicted_label = label_mapping.get(predicted_class_id, "Unknown")
            confidence = np.max(prediction_probs) * 100

            # 5. Display Result
            st.success(f"**Prediction:** {predicted_label}")
            st.info(f"**Confidence:** {confidence:.2f}%")
            
            # Optional: Show probabilities for all classes
            with st.expander("See detailed probabilities"):
                st.bar_chart(pd.DataFrame(prediction_probs.T, index=label_mapping.values(), columns=["Probability"]))