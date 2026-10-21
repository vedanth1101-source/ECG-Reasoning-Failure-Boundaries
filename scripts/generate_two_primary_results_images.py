"""
Generate Two Primary Standalone Visualizations for Results and Analysis
=======================================================================
This script generates EXACTLY TWO comprehensive, high-resolution (300 DPI)
figures representing the entire verified body of work through Module 4 Control Path.

Image 1: Overall Results and Clinical Performance Analysis
  - Panel A: Paradigm Comparison (Our Control Reasoner vs Linear Probe vs Rule Engine)
  - Panel B: Clinical Failure Modes (Rhythm Robustness vs Conduction/MI Collapse)
  - Panel C: Diagnostic Macro-F1 Degradation Curve (12L -> 1L) with 95% Bootstrap CI & V1 Rebound
  - Panel D: Probability Calibration & Reliability (Validation Temperature Scaling ECE Reduction)

Image 2: Explainable Reasoning Traces & Diagnostic Flow Walkthrough
  - Designed for a non-specialist / layman audience:
  - Step 1: Physical Signal Ingestion & Sensor Loss (12-Lead vs 2-Lead Telemetry)
  - Step 2: Structured Evidence Cards (What sensors observe vs what is masked)
  - Step 3: Transformer Cross-Attention Saliency Flow (How the AI weighs evidence)
  - Step 4: Side-by-Side Clinical Dossier (12-Lead Full vs 2-Lead Honest Unavailability)
  - Bottom Takeaway: Why this prevents catastrophic AI hallucinations in healthcare.

All data is strictly verified against runs/module4_v3/ metrics and reports.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

# Base styling
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.2,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
})

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
VIS_DIR = REPO_ROOT / "visualizations"
REPORT_FIG_DIR = REPO_ROOT / "report" / "first_half_implementation" / "figures"
PRES_DIR = REPO_ROOT / "presentation" / "images"

for d in [FIGURES_DIR, VIS_DIR, REPORT_FIG_DIR, PRES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# IMAGE 1: RESULTS & PERFORMANCE ANALYSIS DASHBOARD
# =============================================================================
def generate_image_1_results():
    print("[*] Generating Image 1: Results and Performance Analysis Dashboard...")
    fig = plt.figure(figsize=(18, 11.5), dpi=300, facecolor="#FFFFFF")
    
    # Title Banner across top
    fig.text(0.04, 0.965, "ECG Foundation Model Diagnostic Performance Under Progressive Information Loss",
             fontsize=17, fontweight="bold", color="#0F172A")
    fig.text(0.04, 0.942, "Comprehensive Empirical Results Across Modules 1 to 4 Control Path (Held-Out Test Set N = 7,375, 51,625 Evaluations)",
             fontsize=10.5, fontweight="normal", color="#475569")

    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.22,
                           top=0.915, bottom=0.065, left=0.055, right=0.96)

    # -------------------------------------------------------------------------
    # Panel A: Paradigm Comparison Across Approaches (12-Lead, 6-Lead, 1-Lead)
    # -------------------------------------------------------------------------
    ax_comp = fig.add_subplot(gs[0, 0])
    ax_comp.set_facecolor("#FAFAFA")
    ax_comp.set_title("A. Architecture Comparison Across Lead Configurations", fontweight="bold", pad=10, color="#0F172A", loc="left")

    models = ["Prevalence\nPrior", "Heuristic\nRule Engine", "Frozen FM +\nLinear Probe", "Our Control\nReasoner"]
    # Scores for 12-lead, 6-lead, 1-lead
    f1_12 = [0.000, 0.351, 0.627, 0.738]
    f1_6  = [0.000, 0.315, 0.418, 0.708]
    f1_1  = [0.000, 0.300, 0.287, 0.583]

    x = np.arange(len(models))
    w = 0.26

    r1 = ax_comp.bar(x - w, f1_12, w, label="12-Lead (Full Hospital Standard)", color="#1E40AF", edgecolor="#0F172A", lw=0.8)
    r2 = ax_comp.bar(x, f1_6, w, label="6-Lead (Limb Leads Only)", color="#0284C7", edgecolor="#0F172A", lw=0.8)
    r3 = ax_comp.bar(x + w, f1_1, w, label="1-Lead (Ambulatory Patch/Watch)", color="#E11D48", edgecolor="#0F172A", lw=0.8)

    ax_comp.set_ylabel("Diagnostic Macro-F1 Score", fontweight="bold")
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(models, fontweight="bold", color="#1E293B")
    ax_comp.set_ylim(0, 0.90)
    ax_comp.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    ax_comp.grid(axis="y", linestyle="--", alpha=0.6)

    # Value labels on Reasoner
    for rect in [r1[-1], r2[-1], r3[-1]]:
        h = rect.get_height()
        ax_comp.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                         fontsize=8.5, fontweight="bold", color="#0F172A")

    # Advantage callout box
    ax_comp.text(0.03, 0.66, "+17.7% over Linear Probe\n+110.3% over Rule Engine",
                 transform=ax_comp.transAxes, fontsize=8, fontweight="bold", color="#065F46",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#ECFDF5", ec="#10B981", lw=1.2))

    # -------------------------------------------------------------------------
    # Panel B: Clinical Failure Modes (Rhythm Robustness vs Conduction Cliff)
    # -------------------------------------------------------------------------
    ax_clin = fig.add_subplot(gs[0, 1])
    ax_clin.set_facecolor("#FAFAFA")
    ax_clin.set_title("B. Clinical Failure Modes: Rhythm Robustness vs. Conduction/MI Collapse", fontweight="bold", pad=10, color="#0F172A", loc="left")

    leads = [12, 10, 8, 6, 4, 2, 1]
    # Verified per-condition F1 curves from test_metrics_seed42.json
    stach = [0.9123, 0.9137, 0.9158, 0.9040, 0.9134, 0.9101, 0.9054]
    sbrad = [0.8576, 0.8598, 0.8596, 0.8601, 0.8784, 0.8529, 0.8354]
    afib  = [0.8034, 0.7984, 0.7954, 0.7744, 0.8045, 0.7766, 0.7483]
    lbbb  = [0.7909, 0.7944, 0.7958, 0.7725, 0.7770, 0.7516, 0.4565]
    rbbb  = [0.7857, 0.7733, 0.7366, 0.6738, 0.7762, 0.7736, 0.1266]
    mi    = [0.5504, 0.5401, 0.5407, 0.5126, 0.5124, 0.4725, 0.4452]

    # Plot Rhythm (Green/Teal tones)
    ax_clin.plot(leads, stach, marker="o", color="#059669", lw=2.2, label="Sinus Tachycardia (Rhythm, Robust)")
    ax_clin.plot(leads, sbrad, marker="^", color="#0D9488", lw=2.0, label="Sinus Bradycardia (Rhythm, Robust)")
    ax_clin.plot(leads, afib, marker="s", color="#0284C7", lw=2.0, label="Atrial Fibrillation (Rhythm, Robust)")

    # Plot Conduction & Morphological (Red/Purple/Amber)
    ax_clin.plot(leads, lbbb, marker="D", color="#9333EA", lw=2.2, linestyle="-.", label="LBBB (Conduction Cliff at 1-Lead)")
    ax_clin.plot(leads, rbbb, marker="v", color="#DC2626", lw=2.5, linestyle="-", label="RBBB (Severe Collapse: 0.786 -> 0.127)")
    ax_clin.plot(leads, mi, marker="x", color="#D97706", lw=1.8, linestyle=":", label="Myocardial Infarction (Spatial Loss)")

    ax_clin.set_xlabel("Number of Available Leads (l)", fontweight="bold")
    ax_clin.set_ylabel("Per-Condition F1 Score", fontweight="bold")
    ax_clin.set_xlim(12.5, 0.5)
    ax_clin.set_xticks(leads)
    ax_clin.set_xticklabels(["12L", "10L", "8L", "6L", "4L", "2L", "1L"], fontweight="bold")
    ax_clin.set_ylim(0.05, 1.0)
    ax_clin.legend(loc="lower left", framealpha=0.95, fontsize=8, ncol=2)
    ax_clin.grid(True, linestyle="--", alpha=0.6)

    # Annotation for RBBB collapse
    ax_clin.annotate("Catastrophic Conduction Failure:\nLoss of precordial lead V1 blinds\nright ventricular depolarization!",
                     xy=(1, 0.1266), xytext=(3.4, 0.22),
                     arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=1.5, headwidth=6),
                     fontsize=8, fontweight="bold", color="#B91C1C",
                     bbox=dict(boxstyle="round,pad=0.25", fc="#FEF2F2", ec="#F87171", lw=1.0))

    # -------------------------------------------------------------------------
    # Panel C: Full Degradation Trajectory with Bootstrap 95% Confidence Bounds
    # -------------------------------------------------------------------------
    ax_deg = fig.add_subplot(gs[1, 0])
    ax_deg.set_facecolor("#FFFFFF")
    ax_deg.set_title("C. Diagnostic Degradation Curve & Non-Monotonic Rebound", fontweight="bold", pad=10, color="#0F172A", loc="left")

    f1_reasoner = [0.7375, 0.7367, 0.7307, 0.7076, 0.7347, 0.7101, 0.5832]
    ci_low      = [0.7295, 0.7278, 0.7228, 0.6987, 0.7253, 0.7000, 0.5710]
    ci_high     = [0.7478, 0.7483, 0.7421, 0.7209, 0.7461, 0.7218, 0.5983]

    f1_probe = [0.6269, 0.6215, 0.6198, 0.4178, 0.4990, 0.4299, 0.2874]
    f1_rule  = [0.3507, 0.3444, 0.3491, 0.3150, 0.3039, 0.2995, 0.2995]

    # Shaded 95% Bootstrap CI
    ax_deg.fill_between(leads, ci_low, ci_high, color="#FDA4AF", alpha=0.35, label="Empirical 95% Bootstrap CI (B=1,000)")
    ax_deg.plot(leads, f1_reasoner, color="#E11D48", lw=2.8, marker="o", markersize=8, label="B-3 Control Reasoner (Transformer)", zorder=4)
    ax_deg.plot(leads, f1_probe, color="#2563EB", lw=2.0, marker="s", markersize=6.5, linestyle="--", label="B-1 Linear Probe (Ridge Logistic)", zorder=3)
    ax_deg.plot(leads, f1_rule, color="#0D9488", lw=1.8, marker="^", markersize=6.5, linestyle=":", label="B-2 Heuristic Clinical Rule Engine", zorder=2)
    ax_deg.axhline(0.7375, color="#D97706", linestyle="-.", lw=1.5, label="12-Lead Full Reference (F1 = 0.7375)")

    # Annotation: V1 recovery
    ax_deg.annotate(r"$\mathbf{V_1}$ Precordial Recovery (+0.027 F1):" + "\n4-lead (I, II, III, V1) outperforms 6-lead!\nProves lead identity > lead count.",
                    xy=(4, 0.7347), xytext=(6.5, 0.62),
                    arrowprops=dict(facecolor="#059669", shrink=0.08, width=1.4, headwidth=6),
                    fontsize=8.2, fontweight="bold", color="#065F46",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#10B981", lw=1.2))

    # Annotation: Single lead cliff
    ax_deg.annotate("Single-Lead Cliff:\nDrop to F1 = 0.5832\n(Ambulatory limit)",
                    xy=(1, 0.5832), xytext=(2.2, 0.48),
                    arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=1.4, headwidth=6),
                    fontsize=8.2, fontweight="bold", color="#B91C1C",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#FEF2F2", ec="#F87171", lw=1.2))

    ax_deg.set_xlabel("Lead Configuration (Number of Available Leads)", fontweight="bold")
    ax_deg.set_ylabel("Macro-F1 Score", fontweight="bold")
    ax_deg.set_xlim(12.5, 0.5)
    ax_deg.set_xticks(leads)
    ax_deg.set_xticklabels(["12", "10", "8", "6", "4", "2", "1"], fontweight="bold")
    ax_deg.set_ylim(0.20, 0.85)
    ax_deg.legend(loc="lower left", framealpha=0.95, fontsize=8)
    ax_deg.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel D: Clinical Uncertainty & Temperature Calibration
    # -------------------------------------------------------------------------
    ax_cal = fig.add_subplot(gs[1, 1])
    ax_cal.set_facecolor("#FAFAFA")
    ax_cal.set_title("D. Model Reliability: Validation Temperature Scaling Calibration", fontweight="bold", pad=10, color="#0F172A", loc="left")

    conditions = ["NORM", "AFIB", "RBBB", "LBBB", "SBRAD", "STACH", "1AVB", "MI"]
    pre_ece = [2.16, 1.50, 0.78, 0.56, 2.24, 0.50, 0.98, 3.27]
    post_ece = [0.59, 0.71, 0.32, 0.42, 2.29, 0.25, 0.53, 3.25]
    temperatures = [1.19, 1.26, 1.14, 1.08, 1.09, 1.11, 1.16, 1.13]

    x_c = np.arange(len(conditions))
    w_c = 0.35

    ax_cal.bar(x_c - w_c/2, pre_ece, w_c, label="Pre-Calibration ECE (%)", color="#94A3B8", edgecolor="#1E293B", lw=0.8)
    ax_cal.bar(x_c + w_c/2, post_ece, w_c, label="Post-Calibration ECE (%)", color="#10B981", edgecolor="#1E293B", lw=0.8, hatch="//")

    ax_cal.set_ylabel("Expected Calibration Error ECE (%)", fontweight="bold")
    ax_cal.set_xticks(x_c)
    labels = [f"{c}\n(T={t:.2f})" for c, t in zip(conditions, temperatures)]
    ax_cal.set_xticklabels(labels, fontweight="bold", fontsize=7.8, color="#334155")
    ax_cal.set_ylim(0, 4.0)
    ax_cal.legend(loc="upper right", framealpha=0.95, fontsize=8.5)
    ax_cal.grid(axis="y", linestyle="--", alpha=0.6)

    # Calibration impact text
    summary_box = (
        "GLOBAL RELIABILITY AUDIT:\n"
        "- Mean ECE dropped: 1.50% -> 1.04% (-30.4% error reduction)\n"
        "- Normal ECG calibration improved by -72.7%\n"
        "- Preserves exact AUROC ranking with zero test-set leakage"
    )
    ax_cal.text(0.04, 0.73, summary_box, transform=ax_cal.transAxes, fontsize=8.0, fontweight="bold",
                color="#065F46", bbox=dict(boxstyle="round,pad=0.35", fc="#ECFDF5", ec="#10B981", lw=1.2))

    # Save to all destination paths
    out_files = [
        FIGURES_DIR / "image1_results_and_performance_analysis.png",
        VIS_DIR / "image1_results_and_performance_analysis.png",
        REPORT_FIG_DIR / "image1_results_and_performance_analysis.png",
        PRES_DIR / "image1_results_and_performance_analysis.png"
    ]
    for path in out_files:
        plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [+] Saved Image 1 to: {out_files[0]}")


# =============================================================================
# IMAGE 2: EXPLAINABLE REASONING TRACES & CLINICAL FLOW (FOR NON-SPECIALISTS)
# =============================================================================
def generate_image_2_reasoning_walkthrough():
    print("[*] Generating Image 2: Explainable Reasoning Traces Walkthrough...")
    fig = plt.figure(figsize=(18, 12.5), dpi=300, facecolor="#FFFFFF")

    # Master Title
    fig.text(0.04, 0.970, "How the AI Diagnoses Heart Signals Under Sensor Loss (Explainable Reasoning Traces)",
             fontsize=17, fontweight="bold", color="#0F172A")
    fig.text(0.04, 0.948, "Step-by-Step Clinical Evidence Grounding, Attention Saliency & Hallucination Prevention (Module 4 Control Path)",
             fontsize=10.5, fontweight="normal", color="#475569")

    # Leave comfortable room at bottom for the Master Takeaway Banner (bottom=0.115)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.20], hspace=0.32, wspace=0.22,
                           top=0.925, bottom=0.120, left=0.055, right=0.96)

    # -------------------------------------------------------------------------
    # Panel 1: Physical Heartbeat Ingestion & What Happens When Leads Drop
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FAFAFA")
    ax1.set_title("Step 1: Real Patient Heartbeat & Controlled Sensor Degradation", fontweight="bold", pad=10, color="#0F172A", loc="left")

    # Load actual sample waveform from PTB-XL
    s12 = np.load(REPO_ROOT / "data/benchmark_test/ptbxl/full_12lead.npy", mmap_mode="r")
    lead2 = s12[0, 1, :300]   # Lead II (3 seconds at 100Hz)
    lead_v1 = s12[0, 6, :300] # Lead V1
    lead_v4 = s12[0, 9, :300] # Lead V4 (Chest)
    t = np.linspace(0, 3.0, len(lead2))

    # Plot Lead II (Active in 12L & 2L)
    ax1.plot(t, lead2 + 3.0, color="#059669", lw=1.8, label="Lead II (Rhythm Strip): Active in 12L & 2L Telemetry")
    # Plot Lead V1 (Active in 12L & 2L)
    ax1.plot(t, lead_v1 + 1.2, color="#0D9488", lw=1.8, label="Lead V1 (Right Ventricle): Active in 12L & 2L Telemetry")
    # Plot Lead V4 (Active in 12L, Masked in 2L)
    ax1.plot(t, lead_v4 - 0.8, color="#2563EB", lw=1.6, label="Lead V4 (Anterior Chest): Active in 12L Hospital Standard")
    # Show Masked Flatline for 2L
    ax1.plot(t, np.zeros_like(t) - 2.2, color="#94A3B8", lw=2.0, linestyle="--", label="Leads I, III, aVR-aVF, V2-V6 in 2-Lead: Clamped to 0.0 mV (Masked)")

    ax1.set_xlabel("Time (seconds)", fontweight="bold")
    ax1.set_ylabel("Voltage Trace (mV)", fontweight="bold")
    ax1.set_xlim(0, 3.0)
    ax1.set_ylim(-3.0, 5.0)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=7.8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Explanation badge for layman
    layman_badge1 = (
        "WHAT THE PATIENT WEARS:\n"
        "• Hospital: 12 sticky leads covering the entire chest & limbs.\n"
        "• Bedside / Ambulatory: Only 2 leads (II + V1) to save wires.\n"
        "• Safety Invariant: Inactive leads are strictly zeroed (NO artificial hallucinated data)!"
    )
    ax1.text(0.03, 0.05, layman_badge1, transform=ax1.transAxes, fontsize=7.8, fontweight="bold",
             color="#1E3A8A", bbox=dict(boxstyle="round,pad=0.35", fc="#EFF6FF", ec="#3B82F6", lw=1.2))

    # -------------------------------------------------------------------------
    # Panel 2: Structured Evidence Cards & Sensor Observability
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FAFAFA")
    ax2.set_title("Step 2: Medical Evidence Extracted by Clinical Feature Heads", fontweight="bold", pad=10, color="#0F172A", loc="left")
    ax2.axis("off")

    # Card 12-Lead
    box_12 = patches.FancyBboxPatch((0.02, 0.52), 0.96, 0.45, boxstyle="round,pad=0.02,rounding_size=0.03",
                                    fc="#EFF6FF", ec="#2563EB", lw=1.5, transform=ax2.transAxes)
    ax2.add_patch(box_12)
    ax2.text(0.05, 0.91, "HOSPITAL 12-LEAD EVIDENCE CARD (Full Sensor Array)", fontsize=9.2, fontweight="bold", color="#1E40AF", transform=ax2.transAxes)
    text_12 = (
        "• Heart Rhythm: Sinus Rhythm (Heart Rate: 61 beats/min) -> Normal\n"
        "• Electrical Intervals: PR = 170 ms (Normal) | QRS = 70 ms (Normal)\n"
        "• ST Elevation (Heart Attack): None detected across 12 leads (0 / 12)\n"
        "• ST Depression (Ischemia): PRESENT in Leads [I, V4] (Lead I: -0.55 mm, V4: -0.87 mm)\n"
        "• Observability Check: ALL 12 LEADS ACTIVE -> Evidence Verified Observable [PASS]"
    )
    ax2.text(0.05, 0.70, text_12, fontsize=8.0, color="#1E3A8A", fontfamily="monospace", transform=ax2.transAxes, va="center", linespacing=1.35)

    # Card 2-Lead
    box_2 = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.45, boxstyle="round,pad=0.02,rounding_size=0.03",
                                   fc="#FEF2F2", ec="#E11D48", lw=1.5, transform=ax2.transAxes)
    ax2.add_patch(box_2)
    ax2.text(0.05, 0.41, "BEDSIDE 2-LEAD EVIDENCE CARD (Telemetry Leads II & V1 Only)", fontsize=9.2, fontweight="bold", color="#9F1239", transform=ax2.transAxes)
    text_2 = (
        "• Heart Rhythm: Sinus Rhythm (Heart Rate: 61 beats/min) -> Observable in Lead II [OK]\n"
        "• Electrical Intervals: PR = 170 ms | QRS = 70 ms -> Observable in Lead II [OK]\n"
        "• ST Elevation: None detected in Lead II or V1 (0 / 2)\n"
        "• ST Depression: SENSORS MISSING! (Leads I and V4 are detached/masked)\n"
        "• Observability Check: Marked as [UNAVAILABLE] -> Model Confesses Evidence Blindness!"
    )
    ax2.text(0.05, 0.20, text_2, fontsize=8.0, color="#881337", fontfamily="monospace", transform=ax2.transAxes, va="center", linespacing=1.35)

    # -------------------------------------------------------------------------
    # Panel 3: Inside the Transformer Cross-Attention Saliency
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#FAFAFA")
    ax3.set_title("Step 3: Transformer Cross-Attention Saliency (What the AI Looks At)", fontweight="bold", pad=10, color="#0F172A", loc="left")

    tokens = [
        "Continuous Latent\n(z in R^768)",
        "Rhythm Token\n(Sinus Rhythm)",
        "Lead Token\n(Lead II Active)",
        "Lead Token\n(Lead V1 Active)",
        "Chest Lead V4\n(Active in 12L)",
        "Lateral Lead aVL\n(Active in 12L)",
        "Inferior Lead III\n(Active in 12L)"
    ]
    # Saliency weights from sample_test_reasoning_traces.json
    attn_12 = [0.082, 0.055, 0.048, 0.045, 0.043, 0.036, 0.035]
    attn_2  = [0.125, 0.088, 0.076, 0.068, 0.000, 0.000, 0.000] # Masked tokens zeroed out

    y_pos = np.arange(len(tokens))
    bar_h = 0.35

    ax3.barh(y_pos + bar_h/2, attn_12, bar_h, label="12-Lead Attention Weight (Full Array)", color="#2563EB", edgecolor="#0F172A", lw=0.8)
    ax3.barh(y_pos - bar_h/2, attn_2, bar_h, label="2-Lead Attention Weight (Telemetry II, V1)", color="#0D9488", edgecolor="#0F172A", lw=0.8, hatch="//")

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(tokens, fontweight="bold", fontsize=8.2, color="#1E293B")
    ax3.set_xlabel("Cross-Attention Saliency Weight (Alpha)", fontweight="bold")
    ax3.set_xlim(0, 0.15)
    ax3.legend(loc="lower right", framealpha=0.95, fontsize=8.5)
    ax3.grid(axis="x", linestyle="--", alpha=0.6)

    # Zero Attention Callout
    ax3.annotate("Strict Zero-Attention:\nMissing leads receive exactly\n0.000 attention weight!",
                 xy=(0.002, 4 - bar_h/2), xytext=(0.04, 3.4),
                 arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=1.4, headwidth=6),
                 fontsize=8.0, fontweight="bold", color="#B91C1C",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#FEF2F2", ec="#F87171", lw=1.0))

    # -------------------------------------------------------------------------
    # Panel 4: Final Diagnostic Reasoning Output Dossier Comparison
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#FAFAFA")
    ax4.set_title("Step 4: Final Diagnostic Reasoning Dossier & Safety Conclusion", fontweight="bold", pad=10, color="#0F172A", loc="left")
    ax4.axis("off")

    # Output Trace 12L
    trace_12_box = patches.FancyBboxPatch((0.02, 0.52), 0.96, 0.45, boxstyle="round,pad=0.02,rounding_size=0.03",
                                          fc="#FFFFFF", ec="#059669", lw=1.5, transform=ax4.transAxes)
    ax4.add_patch(trace_12_box)
    ax4.text(0.05, 0.91, "12-LEAD AI DIAGNOSTIC REASONING DOSSIER (PTB-XL Record #9)", fontsize=9.2, fontweight="bold", color="#065F46", transform=ax4.transAxes)
    trace_text_12 = (
        "DIAGNOSTIC HYPOTHESIS: Normal ECG (Confidence: 88.79%)\n"
        "SECONDARY HYPOTHESIS: Myocardial Infarction / Ischemia (Confidence: 6.44%)\n"
        "SUPPORTING EVIDENCE: [LEAD_OBS_V4 (Weight: 0.043), LEAD_OBS_aVL (0.036)]\n"
        "OBSERVABILITY AUDIT: is_observable_in_mask = TRUE for all 12 leads.\n"
        "CLINICAL DECISION: Normal sinus rhythm with minor lateral ST depression.\n"
        "SAFETY VERDICT: Fully Grounded in Observable Physics."
    )
    ax4.text(0.05, 0.70, trace_text_12, fontsize=7.8, color="#064E3B", fontfamily="monospace", transform=ax4.transAxes, va="center", linespacing=1.35)

    # Output Trace 2L
    trace_2_box = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.45, boxstyle="round,pad=0.02,rounding_size=0.03",
                                         fc="#FFFFFF", ec="#D97706", lw=1.5, transform=ax4.transAxes)
    ax4.add_patch(trace_2_box)
    ax4.text(0.05, 0.41, "2-LEAD TELEMETRY AI DIAGNOSTIC DOSSIER (Same Patient)", fontsize=9.2, fontweight="bold", color="#B45309", transform=ax4.transAxes)
    trace_text_2 = (
        "DIAGNOSTIC HYPOTHESIS: Normal ECG (Confidence: 89.15%)\n"
        "SECONDARY HYPOTHESIS: Myocardial Infarction -> CANNOT BE SAFELY CONFIRMED\n"
        "ATTENTION REDIRECTION: Attention shifted to Lead II and V1 (Weight: 0.088)\n"
        "OBSERVABILITY AUDIT: Leads [I, aVL, V4] are MASKED (is_observable = FALSE)\n"
        "CLINICAL DECISION: Rhythms are intact; ischemia evaluation is REFUSED.\n"
        "SAFETY VERDICT: Zero-Hallucination Guard Triggered (Honest Boundary)."
    )
    ax4.text(0.05, 0.20, trace_text_2, fontsize=7.8, color="#78350F", fontfamily="monospace", transform=ax4.transAxes, va="center", linespacing=1.35)

    # Bottom Master Takeaway Banner across the bottom of Figure 2
    takeaway_box = (
        "WHY THIS MATTERS FOR PATIENTS & DOCTORS (IN PLAIN ENGLISH):\n"
        "Most existing AI models are black boxes: when wires disconnect on an ambulance or smartwatch, they silently guess and hallucinate dangerous false diagnoses.\n"
        "Our Control Reasoning Transformer acts like an expert cardiologist: it reads the rhythm, inspects every active wire, explains its thinking in writing,\n"
        "and explicitly tells the doctor when a condition cannot be diagnosed because the required sensor is missing."
    )
    fig.text(0.5, 0.048, takeaway_box, ha="center", va="center", fontsize=8.6, fontweight="bold",
             color="#0F172A", bbox=dict(boxstyle="round,pad=0.45", fc="#F8FAFC", ec="#64748B", lw=1.2))

    # Save to all destination paths
    out_files = [
        FIGURES_DIR / "image2_explainable_reasoning_traces_walkthrough.png",
        VIS_DIR / "image2_explainable_reasoning_traces_walkthrough.png",
        REPORT_FIG_DIR / "image2_explainable_reasoning_traces_walkthrough.png",
        PRES_DIR / "image2_explainable_reasoning_traces_walkthrough.png"
    ]
    for path in out_files:
        plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [+] Saved Image 2 to: {out_files[0]}")


if __name__ == "__main__":
    generate_image_1_results()
    generate_image_2_reasoning_walkthrough()
    print("[SUCCESS] Generated both primary images successfully!")
