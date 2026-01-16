# ==========================================
# 1. IMPORTS & SETUP
# ==========================================
import os
import re
import string
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Natural Language Processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine Learning & Deep Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Download necessary NLTK data (quietly)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("Libraries imported successfully.")

# ==========================================
# 2. DATA LOADING & CONFIGURATION
# ==========================================
# Configuration Constants
DATA_PATH = "mental_health_data.csv" 
VOCAB_SIZE = 8000
MAX_LEN = 100
EMBEDDING_DIM = 128
EPOCHS = 20  
BATCH_SIZE = 32

# Load Data
try:
    df = pd.read_csv(DATA_PATH)
    # Filter only necessary columns
    df = df[['statement', 'status']]
    df.dropna(inplace=True)
    print(f"Data loaded. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: File not found at {DATA_PATH}. Please ensure the CSV file exists.")
    exit()

# ==========================================
# 3. TEXT PREPROCESSING (CLEANING)
# ==========================================
# Define label mapping
label_mapping = {
    'Anxiety': 0,
    'Normal': 1,
    'Depression': 2,
    'Suicidal': 3,
    'Stress': 4,
    'Bipolar': 5,
    'Personality disorder': 6,
    'Personality Disorder': 6 # Handle potential casing variations
}

# Apply mapping
df['status'] = df['status'].map(label_mapping)
# Drop any rows where status mapping failed (became NaN)
df.dropna(subset=['status'], inplace=True)
df['status'] = df['status'].astype(int)

# Cleaning Function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase everything
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    
    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 4. Remove numbers
    text = re.sub(r'\w*\d\w*', '', text)
    
    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("Cleaning text data...")
df['statement'] = df['statement'].apply(clean_text)

# ==========================================
# 4. TOKENIZATION (BPE)
# ==========================================
# We use Byte Pair Encoding (BPE). It breaks words into subwords 
# (e.g., "playing" -> "play" + "ing"), which helps handle unknown words.

# Save text to a temp file for training the tokenizer
tokenizer_corpus_file = 'texts_for_tokenizer.txt'
with open(tokenizer_corpus_file, 'w', encoding='utf-8') as f:
    for text in df['statement'].tolist():
        f.write(text + '\n')

# Initialize Tokenizer
bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
bpe_tokenizer.pre_tokenizer = Whitespace()

# Train Tokenizer
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"], 
    vocab_size=VOCAB_SIZE
)
print("Training BPE tokenizer...")
bpe_tokenizer.train([tokenizer_corpus_file], trainer)

# Process all text
sequences = [bpe_tokenizer.encode(text).ids for text in df['statement'].tolist()]

# Padding: Ensure all sequences are the same length (MAX_LEN)
X = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=MAX_LEN, padding='post', truncating='post'
)
y = df['status'].values

print(f"Tokenization complete. Input shape: {X.shape}")

# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================
# We split the data: 80% for training the model, 20% for testing it.
# stratify=y ensures we keep the same balance of classes in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 6. MODEL BUILDING (LSTM)
# ==========================================
print("Building the LSTM model...")

model = tf.keras.Sequential([
    # Layer 1: Embedding
    # Converts integer token IDs into dense vectors of fixed size.
    # mask_zero=True tells the LSTM to ignore the padding zeros we added.
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True),

    # Layer 2: Bidirectional LSTM
    # Reads the text forwards AND backwards to understand context better.
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),

    # Layer 3: Dropout
    # Randomly turns off neurons during training to prevent overfitting (memorization).
    tf.keras.layers.Dropout(0.5),

    # Layer 4: Dense
    # A standard neural network layer to interpret the LSTM features.
    tf.keras.layers.Dense(64, activation='relu'),

    # Layer 5: Output
    # Uses 'softmax' to give a probability for each of the 7 classes.
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy', # Used when labels are integers (0, 1, 2...)
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 7. TRAINING
# ==========================================
print("Starting training...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)

# ==========================================
# 8. EVALUATION
# ==========================================
print("\nEvaluating model...")

# Generate predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1) # Convert probabilities to class labels

# Reverse mapping to get string labels back
reverse_label_map = {v: k for k, v in label_mapping.items()}
target_names = [reverse_label_map[i] for i in sorted(reverse_label_map.keys()) if i in np.unique(y)]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Save the model
model.save("mental_health_lstm_model.h5")
print("Model saved to mental_health_lstm_model.h5")

# Optional: Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()