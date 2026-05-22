import matplotlib.pyplot as plt
import numpy as np

files = ["file1", "file2", "file3", "file4"]
x = np.arange(len(files))
width = 0.12  # narrower since we have 6 bars per file (3 models × 2 stages)

# PESQ values
pesq_wav2vec_32 = [1.076, 1.054, 1.245, 1.045]
pesq_wav2vec_45 = [1.091, 1.051, 1.336, 1.056]
pesq_hubert_32  = [1.085, 1.129, 1.279, 1.069]
pesq_hubert_45  = [1.126, 1.083, 1.402, 1.078]
pesq_wavlm_32   = [1.101, 1.107, 1.202, 1.072]
pesq_wavlm_45   = [1.098, 1.081, 1.321, 1.076]

# STOI values
stoi_wav2vec_32 = [0.637, 0.533, 0.753, 0.397]
stoi_wav2vec_45 = [0.653, 0.536, 0.739, 0.420]
stoi_hubert_32  = [0.660, 0.619, 0.756, 0.526]
stoi_hubert_45  = [0.712, 0.581, 0.784, 0.553]
stoi_wavlm_32   = [0.647, 0.591, 0.766, 0.556]
stoi_wavlm_45   = [0.685, 0.551, 0.780, 0.522]

# DNSMOS values
dnsmos_wav2vec_32 = [3.063, 2.036, 2.360, 2.824]
dnsmos_wav2vec_45 = [3.022, 2.673, 2.830, 2.767]
dnsmos_hubert_32  = [2.708, 2.577, 2.781, 3.006]
dnsmos_hubert_45  = [2.689, 3.121, 2.463, 3.155]
dnsmos_wavlm_32   = [2.663, 2.653, 2.692, 2.559]
dnsmos_wavlm_45   = [2.677, 2.489, 2.230, 3.109]

# PESQ chart
plt.figure(figsize=(10,6))
plt.bar(x - 2*width, pesq_wav2vec_32, width, label="Wav2Vec2 (32)")
plt.bar(x - width, pesq_wav2vec_45, width, label="Wav2Vec2 (45k)")
plt.bar(x, pesq_hubert_32, width, label="HuBERT (32)")
plt.bar(x + width, pesq_hubert_45, width, label="HuBERT (45k)")
plt.bar(x + 2*width, pesq_wavlm_32, width, label="WavLM (32)")
plt.bar(x + 3*width, pesq_wavlm_45, width, label="WavLM (45k)")
plt.title("PESQ Comparison per File")
plt.xticks(x, files)
plt.legend()
plt.tight_layout()
plt.show()

# STOI chart
plt.figure(figsize=(10,6))
plt.bar(x - 2*width, stoi_wav2vec_32, width, label="Wav2Vec2 (32)")
plt.bar(x - width, stoi_wav2vec_45, width, label="Wav2Vec2 (45k)")
plt.bar(x, stoi_hubert_32, width, label="HuBERT (32)")
plt.bar(x + width, stoi_hubert_45, width, label="HuBERT (45k)")
plt.bar(x + 2*width, stoi_wavlm_32, width, label="WavLM (32)")
plt.bar(x + 3*width, stoi_wavlm_45, width, label="WavLM (45k)")
plt.title("STOI Comparison per File")
plt.xticks(x, files)
plt.legend()
plt.tight_layout()
plt.show()

# DNSMOS chart
plt.figure(figsize=(10,6))
plt.bar(x - 2*width, dnsmos_wav2vec_32, width, label="Wav2Vec2 (32)")
plt.bar(x - width, dnsmos_wav2vec_45, width, label="Wav2Vec2 (45k)")
plt.bar(x, dnsmos_hubert_32, width, label="HuBERT (32)")
plt.bar(x + width, dnsmos_hubert_45, width, label="HuBERT (45k)")
plt.bar(x + 2*width, dnsmos_wavlm_32, width, label="WavLM (32)")
plt.bar(x + 3*width, dnsmos_wavlm_45, width, label="WavLM (45k)")
plt.title("DNSMOS Comparison per File")
plt.xticks(x, files)
plt.legend()
plt.tight_layout()
plt.show()
