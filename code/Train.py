import json, os, re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

MODEL_DIR = "model_files"
MAX_LEN = 200
NUM_WORDS = 10000

def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s

def text_to_sequence(text, word_index, maxlen=MAX_LEN):
    words = clean_text(text).split()
    seq = []
    for w in words:
        idx = word_index.get(w, 2)
        if idx >= NUM_WORDS:
            idx = 2
        seq.append(idx)
    return pad_sequences([seq], maxlen=maxlen, padding='post', truncating='post')

# LOAD ONCE (important)
model = load_model(os.path.join(MODEL_DIR, 'best_model.h5'))

with open(os.path.join(MODEL_DIR, 'word_index.json')) as f:
    word_index = json.load(f)

def predict(text):
    seq = text_to_sequence(text, word_index)
    prob = float(model.predict(seq)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"
    return {"probability": prob, "label": label}