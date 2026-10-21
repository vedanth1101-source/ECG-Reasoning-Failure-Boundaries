"""
Visualizing Module 3 Output: Z_ECG (768-Dimensional Foundation Model Latent Vector)
====================================================================================
Module 3 is the Foundation Model Adaptation & Dual Representation stage:
  - Architecture: Frozen ECG-FM (Wav2Vec2CMSCModel, 90.88M parameters, 12 layers, 12 heads).
  - Output: Z_ECG in R^768 (standardized continuous latent representation pooled from
            the 12th Transformer encoder layer).

Modes:
  1. Default (Fast Precomputed):
     python visualize_z_ecg.py               # Visualizes test benchmark sample #0
     python visualize_z_ecg.py --sample 15   # Visualizes sample #15
     python visualize_z_ecg.py --no-gui      # Headless export to PNG

  2. Live Inference on any ECG .npy file (requires PyTorch, e.g. py -3.11):
     py -3.11 visualize_z_ecg.py --ecg runs/module1_v2/dataset/test/records/CHAPMAN_JS00010.npy
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
})

REPO_ROOT = Path(__file__).resolve().parent
LATENTS_DIR = REPO_ROOT / "data" / "latents" / "ptbxl"
VIS_DIR = REPO_ROOT / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

LEVEL_NAMES = [
    ("12_lead", "12 Leads (C12_FULL)"),
    ("10_lead", "10 Leads (C10_PREC_LIMB)"),
    ("8_lead",  "8 Leads (C8_INDEPENDENT)"),
    ("6_lead",  "6 Leads (C6_LIMB_6)"),
    ("4_lead",  "4 Leads (C4_EINTHOVEN)"),
    ("2_lead",  "2 Leads (C2_TELEMETRY)"),
    ("1_lead",  "1 Lead (C1_SINGLE_LEAD)")
]


def load_cached_latents():
    """Loads precomputed test cohort representations for all 7 degradation levels."""
    latents = {}
    for key, label in LEVEL_NAMES:
        fpath = LATENTS_DIR / f"z_test_ecg_fm_{key}.npy"
        if not fpath.exists():
            fpath = LATENTS_DIR / f"z_{key}.npy"
        if fpath.exists():
            latents[key] = np.load(fpath)
        else:
            raise FileNotFoundError(f"Missing latent file: {fpath}")
    return latents


def run_live_extraction(ecg_path: Path):
    """Runs live FrozenECGFM model extraction across the 7 lead degradation levels."""
    import torch
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.models.ecg_fm_primary import FrozenECGFM
    from src.degradation.lead_masking import LeadMaskingOperator, get_stepwise_schedule

    print(f"[*] Instantiating Frozen ECG-FM (90.88M parameters)...")
    model = FrozenECGFM()
    operator = LeadMaskingOperator()
    schedule = get_stepwise_schedule()

    raw = np.load(ecg_path)
    if raw.ndim == 2 and raw.shape[0] != 12 and raw.shape[1] == 12:
        raw = raw.T  # Transpose to (12, T)
    if raw.shape != (12, 1000):
        raise ValueError(f"ECG array must have shape (12, 1000), got {raw.shape}")
    ecg_12 = raw.astype(np.float32)

    latents = {}
    for key, label in LEVEL_NAMES:
        mask_obj = schedule[key]
        masked = operator.apply(ecg_12, mask_obj)
        z_t = model(masked.signal).squeeze(0).cpu().numpy()  # (768,)
        latents[key] = z_t
    return latents


def compute_pca_numpy(X, n_components=2):
    """Computes exact PCA projection using numpy SVD."""
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    _, s, vt = np.linalg.svd(X_centered, full_matrices=False)
    coords = np.dot(X_centered, vt[:n_components].T)
    var_exp = (s[:n_components] ** 2) / np.sum(s ** 2)
    return coords, vt[:n_components], mean, var_exp


def plot_z_ecg(sample_idx=0, ecg_file=None, show_plot=True, save_path=None):
    if ecg_file:
        ecg_path = Path(ecg_file)
        if not ecg_path.exists():
            raise FileNotFoundError(f"ECG file not found: {ecg_path}")
        print(f"[*] Running LIVE Frozen ECG-FM extraction on file: {ecg_path.name}...")
        sample_latents = run_live_extraction(ecg_path)
        z12_sample = sample_latents["12_lead"]
        title_source = f"Live File: {ecg_path.name}"
    else:
        print(f"[*] Loading Module 3 Z_ECG representations for test sample #{sample_idx}...")
        cohort_latents = load_cached_latents()
        z12_all = cohort_latents["12_lead"]
        n_samples, d_model = z12_all.shape
        if sample_idx >= n_samples:
            raise ValueError(f"sample_idx {sample_idx} exceeds cohort size {n_samples}")
        sample_latents = {k: cohort_latents[k][sample_idx] for k, _ in LEVEL_NAMES}
        z12_sample = sample_latents["12_lead"]
        title_source = f"PTB-XL Test Cohort Record #{sample_idx}"

    d_model = len(z12_sample)
    norm_12 = np.linalg.norm(z12_sample)
    mean_val = np.mean(z12_sample)
    std_val = np.std(z12_sample)

    # Compute cosine similarity and distance across all 7 levels
    sims = []
    dists = []
    stacked_z = []
    for key, label in LEVEL_NAMES:
        z_lvl = sample_latents[key]
        stacked_z.append(z_lvl)
        cos_sim = float(np.dot(z12_sample, z_lvl) / (norm_12 * np.linalg.norm(z_lvl) + 1e-9))
        sims.append(cos_sim)
        dists.append(1.0 - cos_sim)
    stacked_z = np.array(stacked_z)  # (7, 768)

    # Create publication-quality 5-panel visualization
    fig = plt.figure(figsize=(15, 10.5), dpi=150, facecolor="#F8FAFC")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.1, 1.1], hspace=0.38, wspace=0.25)

    # PANEL 1 (Top-Left): 1D Latent Activation Profile
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    dims = np.arange(d_model)
    ax1.plot(dims, z12_sample, color="#1E40AF", lw=1.0, alpha=0.9, label=r"$Z_{\mathrm{ECG}}$ features")
    ax1.fill_between(dims, z12_sample, 0, color="#3B82F6", alpha=0.25)
    ax1.axhline(0, color="#64748B", lw=0.8, linestyle="--")
    ax1.set_title(f"Panel A: 12-Lead $Z_{{\\mathrm{{ECG}}}}$ Latent Activation Profile ({title_source})",
                  fontweight="bold", color="#0F172A", pad=8)
    ax1.set_xlabel("Latent Feature Index (0 to 767)", fontweight="bold")
    ax1.set_ylabel("Activation Magnitude", fontweight="bold")
    ax1.set_xlim(0, 767)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    stats_text = (f"Dimension: 768\n"
                  f"Mean: {mean_val:+.3f}\n"
                  f"Std: {std_val:.3f}\n"
                  f"L2 Norm: {norm_12:.2f}")
    ax1.text(0.97, 0.93, stats_text, transform=ax1.transAxes, ha="right", va="top",
             fontsize=8.5, fontweight="bold", color="#1E293B",
             bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec="#93C5FD", lw=1.0))

    # PANEL 2 (Top-Right): 2D Feature Heatmap Reshape (24 x 32 = 768)
    ax2 = fig.add_subplot(gs[0, 1])
    z_grid = z12_sample.reshape(24, 32)
    vmax = max(abs(np.percentile(z_grid, 98)), abs(np.percentile(z_grid, 2)))
    im2 = ax2.imshow(z_grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax2.set_title(r"Panel B: $Z_{\mathrm{ECG}}$ Spatial Feature Reshape $(24 \times 32 = 768)$",
                  fontweight="bold", color="#0F172A", pad=8)
    ax2.set_xlabel("Feature Sub-band Column (0 to 31)", fontweight="bold")
    ax2.set_ylabel("Feature Group Row (0 to 23)", fontweight="bold")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.035, pad=0.03)
    cbar2.set_label("Feature Activation", fontsize=8.5)

    # PANEL 3 (Middle Full-Width): Latent Stack Across 7 Lead Levels
    ax3 = fig.add_subplot(gs[1, :])
    vmax_stack = max(abs(np.percentile(stacked_z, 99)), abs(np.percentile(stacked_z, 1)))
    im3 = ax3.imshow(stacked_z, aspect="auto", cmap="coolwarm", vmin=-vmax_stack, vmax=vmax_stack)
    ax3.set_title("Panel C: Progressive Information Loss Across 7 Lead Degradation Levels "
                  r"($Z_{\mathrm{ECG}}^{(12)} \to Z_{\mathrm{ECG}}^{(1)}$ in $\mathbb{R}^{768}$)",
                  fontweight="bold", color="#0F172A", pad=8)
    ax3.set_yticks(np.arange(7))
    ax3.set_yticklabels([name for _, name in LEVEL_NAMES], fontweight="bold", color="#1E293B")
    ax3.set_xlabel("Latent Feature Index (0 to 767)", fontweight="bold")
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.015, pad=0.02)
    cbar3.set_label("Latent Value", fontsize=8.5)

    # PANEL 4 (Bottom-Left): Cosine Similarity & Distance Decay Curve
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#FFFFFF")
    x_steps = np.arange(7)
    short_labels = ["12-Lead", "10-Lead", "8-Lead", "6-Lead", "4-Lead", "2-Lead", "1-Lead"]
    
    ax4.plot(x_steps, sims, marker="o", color="#0284C7", lw=2.2, markersize=7,
             label=r"Cosine Similarity to $Z_{12}$")
    ax4.set_ylabel("Cosine Similarity", color="#0284C7", fontweight="bold")
    ax4.tick_params(axis="y", labelcolor="#0284C7")
    ax4.set_ylim(min(sims) - 0.05, 1.03)
    ax4.set_xticks(x_steps)
    ax4.set_xticklabels(short_labels, rotation=15, ha="right", fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6)

    ax4_twin = ax4.twinx()
    ax4_twin.bar(x_steps, dists, width=0.35, color="#EF4444", alpha=0.3,
                 edgecolor="#DC2626", lw=1.2, label=r"Cosine Distance $D_l$")
    ax4_twin.set_ylabel("Cosine Distance ($D_l$)", color="#DC2626", fontweight="bold")
    ax4_twin.tick_params(axis="y", labelcolor="#DC2626")
    ax4_twin.set_ylim(0.0, max(dists) * 1.5 + 0.05)
    ax4_twin.grid(False)
    ax4.set_title("Panel D: Latent Drift Relative to 12-Lead Baseline", fontweight="bold", color="#0F172A", pad=8)
    ax4_twin.annotate(f"Max Drift\n$D_1={dists[-1]:.3f}$",
                      xy=(6, dists[-1]), xytext=(4.8, dists[-1] * 1.2 + 0.02),
                      arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
                      fontsize=8.5, fontweight="bold", color="#B91C1C",
                      bbox=dict(boxstyle="round,pad=0.2", fc="#FEF2F2", ec="#FCA5A5"))

    # PANEL 5 (Bottom-Right): 2D Latent Manifold Projection
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#FFFFFF")
    cached = load_cached_latents()
    z12_cohort = cached["12_lead"][:500]
    coords_12, vt, mean_vec, var_exp = compute_pca_numpy(z12_cohort, n_components=2)
    coords_2 = np.dot(cached["2_lead"][:500] - mean_vec, vt.T)
    coords_1 = np.dot(cached["1_lead"][:500] - mean_vec, vt.T)

    ax5.scatter(coords_12[:, 0], coords_12[:, 1], c="#2563EB", alpha=0.5, s=18, label="12 Leads (Complete Baseline)")
    ax5.scatter(coords_2[:, 0], coords_2[:, 1], c="#0D9488", alpha=0.5, s=18, label="2 Leads (Telemetry II, V1)")
    ax5.scatter(coords_1[:, 0], coords_1[:, 1], c="#DC2626", alpha=0.5, s=18, label="1 Lead (Ambulatory II)")

    # Overlay current sample
    cur_12 = np.dot(z12_sample - mean_vec, vt.T)
    cur_1 = np.dot(sample_latents["1_lead"] - mean_vec, vt.T)
    ax5.scatter(cur_12[0], cur_12[1], c="#1E3A8A", edgecolors="#FFFFFF", s=120, marker="*", lw=1.5, zorder=5, label="Current Record (12L)")
    ax5.scatter(cur_1[0], cur_1[1], c="#991B1B", edgecolors="#FFFFFF", s=120, marker="X", lw=1.5, zorder=5, label="Current Record (1L)")
    ax5.annotate("", xy=(cur_1[0], cur_1[1]), xytext=(cur_12[0], cur_12[1]),
                 arrowprops=dict(arrowstyle="->", color="#334155", lw=1.8, linestyle="--"))

    ax5.set_title("Panel E: Latent Cohort Manifold Projection (PCA, $N=500$)", fontweight="bold", color="#0F172A", pad=8)
    ax5.set_xlabel(f"PC 1 ({var_exp[0]*100:.1f}%)", fontweight="bold")
    ax5.set_ylabel(f"PC 2 ({var_exp[1]*100:.1f}%)", fontweight="bold")
    ax5.legend(loc="upper right", framealpha=0.9, fontsize=7.5)
    ax5.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"MODULE 3 OUTPUT: Standardized Continuous Representation $Z_{{\\mathrm{{ECG}}}} \\in \\mathbb{{R}}^{{768}}$\n"
                 f"Frozen ECG-FM Foundation Model (wav2vec2_cmsc, 90.88M parameters, zero trainable adapters)",
                 fontsize=13, fontweight="bold", color="#0F172A", y=0.985)

    if save_path:
        out_file = Path(save_path)
    elif ecg_file:
        out_file = VIS_DIR / f"module3_z_ecg_{Path(ecg_file).stem}.png"
    else:
        out_file = VIS_DIR / f"module3_z_ecg_sample_{sample_idx}.png"

    plt.savefig(out_file, bbox_inches="tight", dpi=200)
    print(f"[+] Saved visualization image to: {out_file}")

    if show_plot:
        print("[*] Displaying interactive plot window (close window to exit)...")
        plt.show()

    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Module 3 Z_ECG Output")
    parser.add_argument("--sample", type=int, default=0, help="Test cohort sample index (default: 0)")
    parser.add_argument("--ecg", type=str, default=None, help="Path to raw ECG .npy waveform array (12, 1000)")
    parser.add_argument("--no-gui", action="store_true", help="Do not open GUI window (headless/save only)")
    parser.add_argument("--output", type=str, default=None, help="Custom output image path")
    args = parser.parse_args()

    out_img = plot_z_ecg(sample_idx=args.sample, ecg_file=args.ecg, show_plot=not args.no_gui, save_path=args.output)
