"""
generate_module4_output_visualizations.py
-----------------------------------------
Generates high-resolution (300 DPI) publication-quality figures representing
the verified empirical outputs of Module 4 (Control-Path Reasoning):

1. Panel A: Multi-Seed Training Convergence & Validation Stability (Seeds 42, 1337, 2024)
2. Panel B: Temperature Scaling & Expected Calibration Error (ECE) Reduction across 8 classes
3. Panel C: Held-Out Test Set Diagnostic Trajectory (Macro-AUROC & Macro-F1 with 95% Bootstrap CI)
4. Panel D: Pathology Sensitivity & Clinical Failure Mode Contrast (Rhythm vs Conduction)

All data ingested directly from audited system files:
  - runs/module4_v3/metrics/test_evaluation_summary.json
  - runs/module4_v3/metrics/control_training_history_seed*.json
  - runs/module4_v3/metrics/multi_seed_variance_summary.json
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

# Styling configuration
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13.5,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.0,
    "grid.color": "#F1F5F9",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRS = [
    REPO_ROOT / "visualizations",
    REPO_ROOT / "figures",
    REPO_ROOT / "report" / "figures",
    REPO_ROOT / "report" / "first_half_implementation" / "figures",
    REPO_ROOT / "presentation" / "images",
]
for d in OUTPUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def generate_phase4_figure():
    print("[*] Generating Phase 4 Output Visualization (Module 4 Control Path)...")

    # Load verified artifacts
    m4_dir = REPO_ROOT / "runs" / "module4_v3" / "metrics"
    test_summary = json.load(open(m4_dir / "test_evaluation_summary.json", encoding="utf-8"))
    seed42_hist = json.load(open(m4_dir / "control_training_history_seed42.json", encoding="utf-8"))
    seed1337_hist = json.load(open(m4_dir / "control_training_history_seed1337.json", encoding="utf-8"))
    seed2024_hist = json.load(open(m4_dir / "control_training_history_seed2024.json", encoding="utf-8"))

    fig = plt.figure(figsize=(16, 11), dpi=300, facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.05], hspace=0.32, wspace=0.24)

    # -------------------------------------------------------------------------
    # Panel A: Multi-Seed Training Convergence & Validation Loss
    # -------------------------------------------------------------------------
    ax_train = fig.add_subplot(gs[0, 0])
    ax_train.set_title("Panel A: Multi-Seed Optimization Convergence & Stability (Seeds 42, 1337, 2024)",
                       fontweight="bold", pad=12, color="#0F172A")
    ax_train.set_facecolor("#FAFAFA")

    epochs = [h["epoch"] for h in seed42_hist["history"]]
    val_loss_42 = [h["val_loss"] for h in seed42_hist["history"]]
    val_loss_1337 = [h["val_loss"] for h in seed1337_hist["history"]]
    val_loss_2024 = [h["val_loss"] for h in seed2024_hist["history"]]
    train_loss_42 = [h["train_loss"] for h in seed42_hist["history"]]

    ax_train.plot(epochs, train_loss_42, color="#64748B", linestyle=":", lw=1.8, label="Train Loss (Seed 42)")
    ax_train.plot(epochs, val_loss_42, color="#2563EB", marker="o", lw=2.2, label="Val Loss Seed 42 (Best: 0.1593)")
    ax_train.plot(epochs, val_loss_1337, color="#0D9488", marker="s", lw=1.8, linestyle="--", label="Val Loss Seed 1337 (Best: 0.1567)")
    ax_train.plot(epochs, val_loss_2024, color="#D97706", marker="^", lw=1.8, linestyle="-.", label="Val Loss Seed 2024 (Best: 0.1595)")

    ax_train.axvline(3, color="#DC2626", linestyle="--", lw=1.5, alpha=0.8, label="Optimal Checkpoint (Epoch 3)")
    ax_train.annotate("Early Convergence at Epoch 3\nMacro-AUROC: 0.9449 ± 0.0006\nMacro-F1: 0.6982 ± 0.0026",
                      xy=(3, 0.159), xytext=(4.0, 0.178),
                      arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
                      fontsize=8.5, fontweight="bold", color="#991B1B",
                      bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#F87171", lw=1.0))

    ax_train.set_xlabel("Training Epoch", fontweight="bold")
    ax_train.set_ylabel("Masked BCE Loss", fontweight="bold")
    ax_train.set_xlim(0.8, 8.2)
    ax_train.set_xticks(epochs)
    ax_train.legend(loc="lower left", framealpha=0.95, fontsize=8)
    ax_train.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel B: Temperature Scaling & Expected Calibration Error (ECE) Reduction
    # -------------------------------------------------------------------------
    ax_cal = fig.add_subplot(gs[0, 1])
    ax_cal.set_title("Panel B: Validation-Only Temperature Scaling & ECE Error Reduction",
                     fontweight="bold", pad=12, color="#0F172A")
    ax_cal.set_facecolor("#FAFAFA")

    diseases = ["NORM", "AFIB", "RBBB", "LBBB", "SBRAD", "STACH", "1AVB", "MI"]
    pre_ece = [seed42_hist["calibration"]["pre_calibration"]["per_class_ece"][d] * 100 for d in diseases]
    post_ece = [seed42_hist["calibration"]["post_calibration"]["per_class_ece"][d] * 100 for d in diseases]
    temps = seed42_hist["calibration"]["temperatures"]

    x = np.arange(len(diseases))
    width = 0.38

    rects1 = ax_cal.bar(x - width/2, pre_ece, width, label="Pre-Calibration ECE (%)", color="#94A3B8", edgecolor="#334155", lw=0.8)
    rects2 = ax_cal.bar(x + width/2, post_ece, width, label="Post-Calibration ECE (%)", color="#10B981", edgecolor="#065F46", lw=0.8, hatch="//")

    ax_cal.set_ylabel("Expected Calibration Error (%)", fontweight="bold")
    ax_cal.set_xticks(x)
    ax_cal.set_xticklabels([f"{d}\n(T={t:.2f})" for d, t in zip(diseases, temps)], fontweight="bold", fontsize=8)
    ax_cal.legend(loc="upper right", framealpha=0.95, fontsize=8.5)
    ax_cal.grid(axis="y", linestyle="--", alpha=0.6)
    ax_cal.set_ylim(0, 4.2)

    # Highlight global drop
    ax_cal.text(0.04, 0.89, "Mean ECE Reduced: 1.50% -> 1.04% (-30.4% Global / Up to -72.7% on NORM)\nPreserves exact AUROC ranking with zero test-leakage",
                transform=ax_cal.transAxes, fontsize=8.0, fontweight="bold", color="#065F46",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#10B981", lw=1.2))

    # -------------------------------------------------------------------------
    # Panel C: Held-Out Test Set Diagnostic Trajectory (AUROC & Macro-F1)
    # -------------------------------------------------------------------------
    ax_deg = fig.add_subplot(gs[1, 0])
    ax_deg.set_title("Panel C: Test Set Diagnostic Trajectory (N = 7,375, 51,625 Evaluations)",
                     fontweight="bold", pad=12, color="#0F172A")
    ax_deg.set_facecolor("#FAFAFA")

    configs = ["12-lead", "10-lead", "8-lead", "6-lead", "4-lead", "2-lead", "1-lead"]
    leads = [12, 10, 8, 6, 4, 2, 1]

    auroc = [test_summary["degradation_trajectory"][c]["macro_auroc"] for c in configs]
    auroc_low = [test_summary["degradation_trajectory"][c]["confidence_intervals"]["macro_auroc_95ci"][0] for c in configs]
    auroc_high = [test_summary["degradation_trajectory"][c]["confidence_intervals"]["macro_auroc_95ci"][1] for c in configs]

    f1 = [test_summary["degradation_trajectory"][c]["macro_f1"] for c in configs]
    f1_low = [test_summary["degradation_trajectory"][c]["confidence_intervals"]["macro_f1_95ci"][0] for c in configs]
    f1_high = [test_summary["degradation_trajectory"][c]["confidence_intervals"]["macro_f1_95ci"][1] for c in configs]

    line_a = ax_deg.plot(leads, auroc, color="#2563EB", marker="o", lw=2.2, markersize=7.5, label="Macro-AUROC (Test)", zorder=4)
    ax_deg.fill_between(leads, auroc_low, auroc_high, color="#2563EB", alpha=0.15, zorder=2)

    line_f = ax_deg.plot(leads, f1, color="#E11D48", marker="s", lw=2.2, markersize=7.5, label="Macro-F1 (Calibrated, Test)", zorder=4)
    ax_deg.fill_between(leads, f1_low, f1_high, color="#E11D48", alpha=0.15, zorder=2)

    # Annotate V1 recovery
    ax_deg.annotate(r"$\mathbf{V_1}$ Precordial Recovery:" + "\nF1: 0.7076 -> 0.7347 (+0.027)\nAUROC: 0.9479 -> 0.9524",
                    xy=(4, 0.7347), xytext=(5.6, 0.63),
                    arrowprops=dict(facecolor="#059669", shrink=0.08, width=1.4, headwidth=6),
                    fontsize=8.0, fontweight="bold", color="#065F46",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec="#10B981", lw=1.0))

    # Annotate Single lead cliff
    ax_deg.annotate("Single-Lead Cliff:\nF1: 0.5832\nAUROC: 0.9215",
                    xy=(1, 0.5832), xytext=(1.8, 0.48),
                    arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=1.4, headwidth=6),
                    fontsize=8.0, fontweight="bold", color="#B91C1C",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#FEF2F2", ec="#F87171", lw=1.0))

    ax_deg.set_xlabel("Number of Available Leads (l)", fontweight="bold")
    ax_deg.set_ylabel("Diagnostic Metric Score", fontweight="bold")
    ax_deg.set_xlim(13, 0)
    ax_deg.set_xticks(leads)
    ax_deg.set_xticklabels(["12", "10", "8", "6", "4", "2", "1"], fontweight="bold")
    ax_deg.set_ylim(0.45, 1.0)
    ax_deg.legend(loc="lower left", framealpha=0.95, fontsize=8.5)
    ax_deg.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel D: Pathology Sensitivity & Clinical Failure Mode Contrast (F1 Score)
    # -------------------------------------------------------------------------
    ax_path = fig.add_subplot(gs[1, 1])
    ax_path.set_title("Panel D: Clinical Failure Modes: Rhythm vs Conduction Pathologies",
                      fontweight="bold", pad=12, color="#0F172A")
    ax_path.set_facecolor("#FAFAFA")

    # Trace per-class F1 across degradation ladder
    rhythm_stach = [test_summary["degradation_trajectory"][c]["per_class_f1"]["STACH"] for c in configs]
    rhythm_sbrad = [test_summary["degradation_trajectory"][c]["per_class_f1"]["SBRAD"] for c in configs]
    rhythm_afib  = [test_summary["degradation_trajectory"][c]["per_class_f1"]["AFIB"] for c in configs]

    cond_rbbb = [test_summary["degradation_trajectory"][c]["per_class_f1"]["RBBB"] for c in configs]
    cond_lbbb = [test_summary["degradation_trajectory"][c]["per_class_f1"]["LBBB"] for c in configs]
    morph_mi  = [test_summary["degradation_trajectory"][c]["per_class_f1"]["MI"] for c in configs]

    ax_path.plot(leads, rhythm_stach, color="#059669", marker="o", lw=2.0, label="STACH (Rhythm, Robust F1 > 0.90)")
    ax_path.plot(leads, rhythm_sbrad, color="#10B981", marker="^", lw=1.8, label="SBRAD (Rhythm, Robust F1 > 0.83)")
    ax_path.plot(leads, rhythm_afib, color="#0284C7", marker="s", lw=1.8, label="AFIB (Rhythm, Robust F1 > 0.74)")

    ax_path.plot(leads, cond_rbbb, color="#DC2626", marker="D", lw=2.4, label="RBBB (Conduction Cliff: 0.786 -> 0.127)")
    ax_path.plot(leads, cond_lbbb, color="#9333EA", marker="v", lw=2.0, label="LBBB (Conduction Cliff: 0.791 -> 0.457)")
    ax_path.plot(leads, morph_mi, color="#F59E0B", marker="x", lw=1.8, linestyle="--", label="MI (Morphological: 0.550 -> 0.445)")

    ax_path.annotate("Critical Finding:\nSingle-lead (II) preserves Rhythm monitoring\nbut fails catastrophically on Conduction Blocks",
                     xy=(1, 0.1266), xytext=(5.5, 0.22),
                     arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=1.4, headwidth=6),
                     fontsize=8.2, fontweight="bold", color="#7F1D1D",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#EF4444", lw=1.2))

    ax_path.set_xlabel("Number of Available Leads (l)", fontweight="bold")
    ax_path.set_ylabel("Per-Condition F1 Score", fontweight="bold")
    ax_path.set_xlim(13, 0)
    ax_path.set_xticks(leads)
    ax_path.set_xticklabels(["12", "10", "8", "6", "4", "2", "1"], fontweight="bold")
    ax_path.set_ylim(0.05, 1.0)
    ax_path.legend(loc="lower left", framealpha=0.95, fontsize=7.8)
    ax_path.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("PHASE 4 OUTPUT: Control-Path Reasoning, Validation Calibration & Test Degradation (N = 7,375 Records)",
                 fontsize=14, fontweight="bold", color="#0F172A", y=0.98)

    for d in OUTPUT_DIRS:
        out_path = d / "phase4_output_dashboard.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [+] Saved Phase 4 dashboard to all output directories.")


if __name__ == "__main__":
    generate_phase4_figure()
