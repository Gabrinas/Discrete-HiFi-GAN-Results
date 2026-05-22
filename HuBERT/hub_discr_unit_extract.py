import os
import glob
import torch
import torchaudio
import numpy as np
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import joblib

# 1. Load pretrained HuBERT
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to("cuda")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

# 2. Load pretrained k-means (joblib pickle)
kmeans = joblib.load("hubert_base_ls960_L9_km500.bin")

# 3. Function to extract discrete units
def extract_units(wav_path, out_path):
    waveform, sr = torchaudio.load(wav_path)
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.inference_mode():
        outputs = hubert(**inputs.to("cuda"))
    features = outputs.last_hidden_state.cpu().numpy()

    # Flatten to 2D for KMeans
    features_2d = features.reshape(-1, features.shape[-1])

    # Quantize with k-means
    units = kmeans.predict(features_2d)

    np.save(out_path, units)
    print(f"Saved discrete units to {out_path}")

# 4. Batch process all audios in my_audios/
input_folder = "unseen_data_at"
output_folder = "unseen_npy"
os.makedirs(output_folder, exist_ok=True)

for wav_path in glob.glob(os.path.join(input_folder, "*.wav")):
    filename = os.path.splitext(os.path.basename(wav_path))[0]
    out_path = os.path.join(output_folder, f"{filename}.npy")
    extract_units(wav_path, out_path)
