# To train wav2vec model with k = 500 and extract discrete as .npy files

from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch
import librosa
from sklearn.cluster import KMeans
import numpy as np
import os
import joblib  # for saving/loading kmeans

# Load model + processor once
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

audio_dir = "unseen_data_at"
save_dir = "unseen_npy_wav2vec"
os.makedirs(save_dir, exist_ok=True)

files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

# Step 1: Collect features for k-means training
all_features = []
for f in files:
    speech, sr = librosa.load(os.path.join(audio_dir, f), sr=16000)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        hs = model(**inputs).last_hidden_state[0].cpu().numpy()
    all_features.append(hs[::5])  # subsample for efficiency

all_features = np.concatenate(all_features, axis=0)
print("Collected features shape:", all_features.shape)

# Step 2: Train k-means once globally
kmeans = KMeans(n_clusters=500, random_state=0)
kmeans.fit(all_features)
joblib.dump(kmeans, "kmeans_500.pkl")  # save model
print("Finished training k-means with 500 clusters")

# Step 3: Apply k-means to each audio
for f in files:
    speech, sr = librosa.load(os.path.join(audio_dir, f), sr=16000)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        hs = model(**inputs).last_hidden_state[0].cpu().numpy()
    discrete_units = kmeans.predict(hs).astype(np.int16)
    out_path = os.path.join(save_dir, f.replace(".wav", ".npy"))
    np.save(out_path, discrete_units)
    print(f"Saved discrete units to {out_path}")
