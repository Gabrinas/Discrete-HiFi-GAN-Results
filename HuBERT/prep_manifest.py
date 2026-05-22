import os, glob, json, numpy as np
from sklearn.model_selection import train_test_split

audio_dir = "my_audios"
unit_dir = "my_npy_wavlm"

pairs = []
for wav_path in glob.glob(os.path.join(audio_dir, "*.wav")):
    base = os.path.splitext(os.path.basename(wav_path))[0]
    npy_path = os.path.join(unit_dir, f"{base}.npy")
    if os.path.exists(npy_path):
        units = np.load(npy_path).tolist()   # load discrete units
        entry = {"audio": wav_path, "units": units}
        pairs.append(entry)

# Split into train/val (90% train, 10% val)
train, val = train_test_split(pairs, test_size=0.1, random_state=42)

with open("train_wavlm.jsonl", "w") as f:
    for item in train:
        f.write(json.dumps(item) + "\n")

with open("val_wavlm.jsonl", "w") as f:
    for item in val:
        f.write(json.dumps(item) + "\n")

print(f"Saved {len(train)} training and {len(val)} validation entries.")
