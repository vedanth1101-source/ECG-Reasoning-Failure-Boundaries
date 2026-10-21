"""
Generate publication-quality scientific plot of Cosine Similarity and Cosine Distance
across the seven primary lead degradation variants for Frozen ECG-FM.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set clean white background
plt.figure(figsize=(10.5, 6), dpi=300)
fig, ax1 = plt.subplots(figsize=(10.5, 6), dpi=300)

variants = [
    "12 Leads\nC12_FULL",
    "10 Leads\nC10_PREC_LIMB",
    "8 Leads\nC8_INDEPENDENT",
    "6 Leads\nC6_LIMB_6",
    "4 Leads\nC4_EINTHOVEN",
    "2 Leads\nC2_TELEMETRY",
    "1 Lead\nC1_SINGLE_LEAD"
]

mean_cosine_sim = [1.00000, 0.99815, 0.99710, 0.90025, 0.91815, 0.88203, 0.85352]
mean_cosine_dist = [0.00000, 0.00185, 0.00290, 0.09975, 0.08185, 0.11797, 0.14648]
p25_dist = [0.00000, 0.00096, 0.00158, 0.07926, 0.05700, 0.08917, 0.10898]
p75_dist = [0.00000, 0.00238, 0.00362, 0.11151, 0.09265, 0.14334, 0.17334]

x = np.arange(len(variants))

# Distinct color palette
color_sim = '#0b559f'   # Deep academic navy blue
color_dist = '#c93b2b'  # Muted crimson

# Left Axis: Cosine Similarity
line1 = ax1.plot(x, mean_cosine_sim, marker='o', color=color_sim, linewidth=2.8, 
                 markersize=8.5, label='Mean Cosine Similarity to 12-Lead Baseline', zorder=4)

# Dynamic text offsets for Cosine Similarity points
sim_offsets = [(0, 10), (0, 10), (0, 10), (0, 12), (0, 12), (0, 11), (0, 11)]
for i, (xi, val, off) in enumerate(zip(x, mean_cosine_sim, sim_offsets)):
    ax1.annotate(f"{val:.3f}", (xi, val), textcoords="offset points", 
                 xytext=off, ha='center', fontsize=9.5, fontweight='bold', color=color_sim)

ax1.set_xlabel('Lead Degradation Variant (Clinically / Device-Motivated Ladder)', fontsize=11.5, fontweight='bold', labelpad=12)
ax1.set_ylabel('Mean Cosine Similarity', color=color_sim, fontsize=11.5, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_sim, labelsize=10)
ax1.set_ylim(0.80, 1.04)
ax1.set_xticks(x)
ax1.set_xticklabels(variants, fontsize=9.5, fontweight='medium')
ax1.grid(True, linestyle='--', alpha=0.5, color='#cccccc')

# Right Axis: Cosine Distance (D_l = 1 - Cosine Sim)
ax2 = ax1.twinx()
error_low = np.array(mean_cosine_dist) - np.array(p25_dist)
error_high = np.array(p75_dist) - np.array(mean_cosine_dist)

bar2 = ax2.bar(x, mean_cosine_dist, width=0.35, alpha=0.22, color=color_dist, 
               edgecolor=color_dist, linewidth=1.5, label='Mean Cosine Distance D_l (Dispersion: IQR [p25-p75])', zorder=2)
ax2.errorbar(x, mean_cosine_dist, yerr=[error_low, error_high], fmt='none', 
             ecolor=color_dist, elinewidth=1.6, capsize=4.5, capthick=1.6, zorder=3)

# Annotate Distance values
for i, (xi, val) in enumerate(zip(x, mean_cosine_dist)):
    if val > 0.005:
        ax2.annotate(f"D={val:.3f}", (xi, val / 2.0), ha='center', va='center', 
                     fontsize=9, fontweight='bold', color=color_dist,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85))

ax2.set_ylabel('Mean Cosine Distance: D_l = 1 - Cosine Sim', color=color_dist, fontsize=11.5, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_dist, labelsize=10)
ax2.set_ylim(0.0, 0.22)
ax2.grid(False)

# Chart Title & Subtitle
plt.title('Frozen ECG-FM Latent Space Sensitivity Across Lead Degradation Variants\n'
          '(Evaluated across validation cohort; Z_ECG in R^768, 90.8M frozen parameters)', 
          fontsize=12.5, fontweight='bold', pad=15)

# Place legend cleanly at lower left where space is completely empty
lines = line1 + [bar2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left', frameon=True, framealpha=0.95, fontsize=10,
           bbox_to_anchor=(0.02, 0.05))

plt.tight_layout()

# Save in visualizations and artifacts directory
out_vis = Path(r"e:/ECG/visualizations/cosine_similarity_variants.png")
out_vis.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_vis, dpi=300, bbox_inches='tight')

artifact_dir = Path(r"C:/Users/VEDANTH/.gemini/antigravity-ide/brain/f435be23-38db-476c-9b19-2f897089dcc4")
artifact_dir.mkdir(parents=True, exist_ok=True)
out_art = artifact_dir / "cosine_similarity_variants.png"
plt.savefig(out_art, dpi=300, bbox_inches='tight')
print("Successfully regenerated clean chart.")
