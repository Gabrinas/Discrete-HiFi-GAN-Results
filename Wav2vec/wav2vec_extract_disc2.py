# To extract discrete units as .npy using the previous trained wav2vec kmeans

import os
import librosa
import torch
import numpy as np
import joblib
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Load pretrained Wav2Vec2
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

# Load trained k-means
kmeans = joblib.load("kmeans_500.pkl")

# Directories
test_dir = "unseen_data_at"
save_dir = "unseen_npy_wav2vec"
os.makedirs(save_dir, exist_ok=True)

# Process each test audio
for f in os.listdir(test_dir):
    if f.endswith(".wav"):
        speech, sr = librosa.load(os.path.join(test_dir, f), sr=16000)
        inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            hs = model(**inputs).last_hidden_state[0].cpu().numpy()
        discrete_units = kmeans.predict(hs).astype(np.int16)
        out_path = os.path.join(save_dir, f.replace(".wav", ".npy"))
        np.save(out_path, discrete_units)
        print(f"Saved discrete units for {f} -> {out_path}")
