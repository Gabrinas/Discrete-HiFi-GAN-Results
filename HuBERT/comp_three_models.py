import matplotlib.pyplot as plt
import numpy as np

# Files
files = ["file1", "file2", "file3", "file4"]
x = np.arange(len(files))
width = 0.25  # narrower since we have 3 models

# PESQ values
pesq_wav2vec = [1.091, 1.051, 1.336, 1.056]
pesq_hubert  = [1.126, 1.083, 1.402, 1.078]
pesq_wavlm   = [1.098, 1.081, 1.321, 1.076]

# STOI values
stoi_wav2vec = [0.653, 0.536, 0.739, 0.420]
stoi_hubert  = [0.712, 0.581, 0.784, 0.553]
stoi_wavlm   = [0.685, 0.551, 0.780, 0.522]

# DNSMOS values
dnsmos_wav2vec = [3.022, 2.673, 2.830, 2.767]
dnsmos_hubert  = [2.689, 3.121, 2.463, 3.155]
dnsmos_wavlm   = [2.677, 2.489, 2.230, 3.109]

# PESQ chart
plt.figure(figsize=(8,6))
plt.bar(x - width, pesq_wav2vec, width, label="Wav2Vec2")
plt.bar(x, pesq_hubert, width, label="HuBERT")
plt.bar(x + width, pesq_wavlm, width, label="WavLM")
plt.title("PESQ Comparison (45k steps)")
plt.xticks(x, files)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("compare_charts_curves_PESQ.png", dpi=150)


# STOI chart
plt.figure(figsize=(8,6))
plt.bar(x - width, stoi_wav2vec, width, label="Wav2Vec2")
plt.bar(x, stoi_hubert, width, label="HuBERT")
plt.bar(x + width, stoi_wavlm, width, label="WavLM")
plt.title("STOI Comparison (45k steps)")
plt.xticks(x, files)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("compare_charts_curves_STOI.png", dpi=150)


# DNSMOS chart
plt.figure(figsize=(8,6))
plt.bar(x - width, dnsmos_wav2vec, width, label="Wav2Vec2")
plt.bar(x, dnsmos_hubert, width, label="HuBERT")
plt.bar(x + width, dnsmos_wavlm, width, label="WavLM")
plt.title("DNSMOS Comparison (45k steps)")
plt.xticks(x, files)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("compare_charts_curves_DNSMOS.png", dpi=150)

