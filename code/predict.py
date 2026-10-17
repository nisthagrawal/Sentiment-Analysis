import json
import os
import re
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ------------------ SILENCE TF WARNINGS ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ------------------ CONFIG ------------------
MODEL_DIR = Path(__file__).resolve().parent / "model_files"
MAX_LEN = 200
NUM_WORDS = 10000
UNK_TOKEN = 2  # unknown word index

# ------------------ TEXT CLEANING ------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

# ------------------ TEXT → SEQUENCE ------------------
def text_to_sequence(text, word_index):
    words = clean_text(text).split()
    seq = []

    for word in words:
        idx = word_index.get(word, UNK_TOKEN)
        if idx >= NUM_WORDS:
            idx = UNK_TOKEN
        seq.append(idx)

    return pad_sequences([seq], maxlen=MAX_LEN, padding='post', truncating='post')

# ------------------ LOAD MODEL (ONLY ONCE) ------------------
model_path = MODEL_DIR / 'best_model.h5'
word_index_path = MODEL_DIR / 'word_index.json'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

if not os.path.exists(word_index_path):
    raise FileNotFoundError(f"Word index not found at {word_index_path}")

model = load_model(str(model_path))

with open(word_index_path, 'r', encoding='utf-8') as f:
    word_index = json.load(f)

# ------------------ PREDICTION FUNCTION ------------------
def predict(text):
    # Handle empty input
    if not text or text.strip() == "":
        return {"error": "Input text cannot be empty"}

    try:
        seq = text_to_sequence(text, word_index)

        # Model prediction
        prob = float(model.predict(seq, verbose=0)[0][0])

        # Label decision
        label = "Positive" if prob >= 0.5 else "Negative"

        # Confidence formatting
        confidence = round(prob, 4)

        return {
            "input_text": text,
            "probability": confidence,
            "label": label
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ------------------ CLI TEST ------------------
if __name__ == "__main__":
    print("Sentiment Analysis Model Ready")

    while True:
        user_input = input("\nEnter review (or type 'exit'): ")

        if user_input.lower() == "exit":
            print("Exiting...")
            break

        result = predict(user_input)
        print("\nResult:", result)