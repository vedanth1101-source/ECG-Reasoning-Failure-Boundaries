from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the requested file (Module 3 Representation / Z_ECG)
file_path = r"runs/module3_v1/representations/1-lead/train/records/CHAPMAN_JS00001.npy"
data = np.load(file_path)

print("=" * 60)
print(f"File: {file_path}")
print(f"Shape: {data.shape}, Dtype: {data.dtype}")
print(f"Values: min = {data.min():.4f}, max = {data.max():.4f}")
print(f"Mean: {data.mean():.4f}, Std: {data.std():.4f}")
print(f"L2 Norm: {np.linalg.norm(data):.4f}")
print("=" * 60)
print(f"First 10 values:\n{data[:10]}")
print("=" * 60)

if data.ndim == 1 and data.shape[0] == 768:
    # Module 3 Latent Vector (Z_ECG in R^768)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Module 3 Latent Representation: Z_ECG (CHAPMAN_JS00001, 1-Lead)",
                 fontsize=14, fontweight="bold")

    # Panel 1: 1D Line Plot of all 768 features
    axes[0, 0].plot(data, color="#1f77b4", lw=1.2)
    axes[0, 0].fill_between(range(768), data, 0, color="#1f77b4", alpha=0.2)
    axes[0, 0].set_title("1D Feature Profile (768 Dimensions)", fontweight="bold")
    axes[0, 0].set_xlabel("Latent Dimension Index (0 to 767)")
    axes[0, 0].set_ylabel("Activation Value")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: 2D Spatial Heatmap (24 x 32 = 768)
    grid = data.reshape(24, 32)
    vmax = max(abs(data.min()), abs(data.max()))
    im = axes[0, 1].imshow(grid, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title("2D Feature Heatmap (24 x 32 = 768)", fontweight="bold")
    axes[0, 1].set_xlabel("Feature Sub-band Column (0 to 31)")
    axes[0, 1].set_ylabel("Feature Group Row (0 to 23)")
    plt.colorbar(im, ax=axes[0, 1], label="Activation")

    # Panel 3: Value Distribution Histogram
    axes[1, 0].hist(data, bins=50, color="#2ca02c", edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("Latent Values Distribution", fontweight="bold")
    axes[1, 0].set_xlabel("Activation Value")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Compare against 12-lead baseline if available
    f12 = Path(r"runs/module3_v1/representations/12-lead/train/records/CHAPMAN_JS00001.npy")
    if f12.exists():
        z12 = np.load(f12)
        cos_sim = float(np.dot(data, z12) / (np.linalg.norm(data) * np.linalg.norm(z12)))
        cos_dist = 1.0 - cos_sim
        axes[1, 1].plot(z12[:100], label="12-Lead Baseline", color="#1f77b4", lw=1.2)
        axes[1, 1].plot(data[:100], label=f"1-Lead Degraded\nCos Sim: {cos_sim:.4f}\nCos Dist: {cos_dist:.4f}",
                        color="#d62728", lw=1.2, linestyle="--")
        axes[1, 1].set_title("Comparison: 1-Lead vs 12-Lead Baseline (First 100 Dims)", fontweight="bold")
        axes[1, 1].set_xlabel("Latent Dimension Index")
        axes[1, 1].set_ylabel("Activation Value")
        axes[1, 1].legend(loc="upper right", fontsize=8.5)
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("representation_plot.png", dpi=150)
    print("\nSaved visualization to: representation_plot.png")
    plt.show()

elif data.ndim == 2 and data.shape[0] == 12:
    # Standard 12-lead ECG signal (12, 1000)
    time = np.arange(data.shape[1]) / 100.0
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axes = plt.subplots(6, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("12-Lead ECG Signal", fontsize=14, fontweight="bold")
    for i, lead in enumerate(lead_names):
        ax = axes[i % 6, i // 6]
        signal = data[i]
        is_masked = np.all(signal == 0)
        ax.plot(time, signal, color="gray" if is_masked else "#1f77b4")
        ax.set_ylabel(f"{lead} (mV)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        label = "Masked (0 mV)" if is_masked else "Active"
        ax.text(0.95, 0.8, label, transform=ax.transAxes, ha="right",
                fontsize=8, color="red" if is_masked else "green")
    axes[5, 0].set_xlabel("Time (seconds)")
    axes[5, 1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig("ecg_plot.png", dpi=150)
    print("\nSaved ECG plot to: ecg_plot.png")
    plt.show()

