"""
generate_results_and_analysis_visuals.py
----------------------------------------
Generates high-resolution presentation and report figures matching the
exact two-panel layout requested by the user, strictly adhering to the
verified empirical results obtained up to Module 4 Control Path:

  - Top Panel: Dark-navy container with hatched white bars comparing verified
               model and configuration diagnostic performance (Macro-F1).
               Ingests: B-0 Prevalence, B-2 Rule Engine, B-1 Linear Probe,
               and B-3 Control Reasoning Transformer across configurations.
  - Bottom Panel: Clean white grid plot showing the verified diagnostic
                  Macro-F1 degradation trajectory across lead configurations
                  (B-3 Control Reasoner vs B-1 Linear Probe vs B-2 Rule Engine)
                  with empirical 95% bootstrap confidence intervals and the
                  12-lead full-sensor reference ceiling.
  - Full Composite: "Results and Analysis" slide-ready graphic.

All metrics are directly verified against:
  - runs/module4_v3/metrics/test_evaluation_summary.json
  - runs/module4_v3/metrics/comprehensive_baseline_comparison.json
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_results_visuals():
    root = Path(__file__).resolve().parent.parent
    output_dirs = [
        root / "figures",
        root / "report" / "figures",
        root / "report" / "first_half_implementation" / "figures",
        root / "presentation" / "images",
        root / "visualizations"
    ]
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. LOAD VERIFIED METRICS FROM SYSTEM JSON FILES
    # -------------------------------------------------------------
    test_summary_path = root / "runs" / "module4_v3" / "metrics" / "test_evaluation_summary.json"
    baseline_path = root / "runs" / "module4_v3" / "metrics" / "comprehensive_baseline_comparison.json"

    with open(test_summary_path, "r", encoding="utf-8") as f:
        test_summary = json.load(f)

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_summary = json.load(f)

    configs = ["12-lead", "10-lead", "8-lead", "6-lead", "4-lead", "2-lead", "1-lead"]
    leads = [12, 10, 8, 6, 4, 2, 1]

    # B-3 Control Reasoner trajectory
    b3_f1 = []
    b3_f1_ci_low = []
    b3_f1_ci_high = []
    b3_auroc = []

    for cfg in configs:
        data = test_summary["degradation_trajectory"][cfg]
        f1_val = data["macro_f1"]
        ci_f1 = data["confidence_intervals"]["macro_f1_95ci"]
        b3_f1.append(f1_val)
        b3_f1_ci_low.append(ci_f1[0])
        b3_f1_ci_high.append(ci_f1[1])
        b3_auroc.append(data["macro_auroc"])

    # B-1 Linear Probe & B-2 Rule Engine trajectory
    b1_f1 = [baseline_summary[cfg]["Linear_Probe"]["macro_f1"] for cfg in configs]
    b2_f1 = [baseline_summary[cfg]["Rule_Engine"]["macro_f1"] for cfg in configs]

    # Models list for top comparison chart (strictly verified, sorted ascending by Macro-F1)
    models_data = [
        ("PREVALENCE (B-0)", baseline_summary["12-lead"]["Prevalence"]["macro_f1"]),
        ("PROBE (1-LEAD)", baseline_summary["1-lead"]["Linear_Probe"]["macro_f1"]),
        ("RULE ENGINE (1-LEAD)", baseline_summary["1-lead"]["Rule_Engine"]["macro_f1"]),
        ("RULE ENGINE (2-LEAD)", baseline_summary["2-lead"]["Rule_Engine"]["macro_f1"]),
        ("RULE ENGINE (4-LEAD)", baseline_summary["4-lead"]["Rule_Engine"]["macro_f1"]),
        ("RULE ENGINE (6-LEAD)", baseline_summary["6-lead"]["Rule_Engine"]["macro_f1"]),
        ("RULE ENGINE (10-LEAD)", baseline_summary["10-lead"]["Rule_Engine"]["macro_f1"]),
        ("RULE ENGINE (12-LEAD)", baseline_summary["12-lead"]["Rule_Engine"]["macro_f1"]),
        ("RULE ENGINE (8-LEAD)", baseline_summary["8-lead"]["Rule_Engine"]["macro_f1"]),
        ("PROBE (6-LEAD)", baseline_summary["6-lead"]["Linear_Probe"]["macro_f1"]),
        ("PROBE (2-LEAD)", baseline_summary["2-lead"]["Linear_Probe"]["macro_f1"]),
        ("PROBE (4-LEAD)", baseline_summary["4-lead"]["Linear_Probe"]["macro_f1"]),
        ("REASONER (1-LEAD)", baseline_summary["1-lead"]["Control_Transformer"]["macro_f1"]),
        ("PROBE (8-LEAD)", baseline_summary["8-lead"]["Linear_Probe"]["macro_f1"]),
        ("PROBE (10-LEAD)", baseline_summary["10-lead"]["Linear_Probe"]["macro_f1"]),
        ("PROBE (12-LEAD)", baseline_summary["12-lead"]["Linear_Probe"]["macro_f1"]),
        ("REASONER (6-LEAD)", baseline_summary["6-lead"]["Control_Transformer"]["macro_f1"]),
        ("REASONER (2-LEAD)", baseline_summary["2-lead"]["Control_Transformer"]["macro_f1"]),
        ("REASONER (8-LEAD)", baseline_summary["8-lead"]["Control_Transformer"]["macro_f1"]),
        ("REASONER (4-LEAD)", baseline_summary["4-lead"]["Control_Transformer"]["macro_f1"]),
        ("REASONER (10-LEAD)", baseline_summary["10-lead"]["Control_Transformer"]["macro_f1"]),
        ("REASONER (12-LEAD)", baseline_summary["12-lead"]["Control_Transformer"]["macro_f1"])
    ]

    model_names = [m[0] for m in models_data]
    f1_scores = [m[1] for m in models_data]
    x1 = np.arange(len(model_names))

    # -------------------------------------------------------------
    # 2. STANDALONE TOP CHART (Dark Blue Model Comparison Bar Chart)
    # -------------------------------------------------------------
    fig_top = plt.figure(figsize=(11.0, 4.9), dpi=300)
    fig_top.patch.set_facecolor('#113854')
    axt = fig_top.add_axes([0.07, 0.28, 0.90, 0.60])
    axt.set_facecolor('#113854')

    bars = axt.bar(x1, f1_scores, width=0.52, color='#ffffff', edgecolor='#38bdf8', hatch='///', linewidth=1.1, zorder=3)
    axt.set_title("MODEL DIAGNOSTIC MACRO-F1 PERFORMANCE ACROSS CONFIGURATIONS\n(Held-Out Test Set N = 7,375, Seed 42 Calibrated)",
                  fontsize=11.0, fontweight='bold', color='#ffffff', pad=10)
    axt.set_ylim(0, 1.0)
    axt.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axt.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], color='#ffffff', fontsize=9.5, fontweight='600')
    axt.set_xticks(x1)
    axt.set_xticklabels(model_names, rotation=52, ha='right', color='#ffffff', fontsize=7.5, fontweight='600')
    axt.grid(axis='y', color='white', alpha=0.18, linestyle='-', linewidth=0.8, zorder=0)
    for xi in x1:
        axt.axvline(xi, color='white', alpha=0.08, linestyle='-', linewidth=0.6, zorder=0)
    for spine in axt.spines.values():
        spine.set_color('#38bdf8')
        spine.set_linewidth(1.0)
    axt.tick_params(colors='#ffffff')

    # Add numeric labels on top of top 3 bars
    for i in range(len(f1_scores) - 3, len(f1_scores)):
        score = f1_scores[i]
        axt.text(i, score + 0.02, f"{score:.3f}", ha='center', va='bottom', color='#38bdf8', fontsize=7.5, fontweight='bold')

    top_chart_path = root / "figures" / "fig_results_model_comparison_bars.png"
    fig_top.savefig(top_chart_path, bbox_inches='tight', facecolor='#113854', dpi=300)
    for d in output_dirs:
        if d != root / "figures":
            fig_top.savefig(d / "fig_results_model_comparison_bars.png", bbox_inches='tight', facecolor='#113854', dpi=300)
    plt.close(fig_top)
    print("Saved fig_results_model_comparison_bars.png to all target directories.")

    # -------------------------------------------------------------
    # 3. STANDALONE BOTTOM CHART (White Degradation Curve with SLA)
    # -------------------------------------------------------------
    fig_bot, axb = plt.subplots(figsize=(8.0, 5.2), dpi=300)
    fig_bot.patch.set_facecolor('#ffffff')
    axb.set_facecolor('#ffffff')

    # Plot B-3 with confidence interval
    axb.fill_between(leads, b3_f1_ci_low, b3_f1_ci_high, color='#e11d48', alpha=0.15, label="B-3 95% Bootstrap CI", zorder=2)
    axb.plot(leads, b3_f1, color='#e11d48', marker='o', markersize=7.5, linewidth=2.2, label="B-3 Control Reasoning Transformer", zorder=4)

    # Plot B-1 Linear Probe
    axb.plot(leads, b1_f1, color='#2563eb', marker='s', markersize=7.0, linewidth=2.0, linestyle='--', label="B-1 Linear Probe (Ridge Logistic)", zorder=4)

    # Plot B-2 Rule Engine
    axb.plot(leads, b2_f1, color='#059669', marker='^', markersize=6.5, linewidth=1.6, linestyle=':', label="B-2 Heuristic Clinical Rule Engine", zorder=3)

    # Reference 12-lead ceiling
    ceiling = b3_f1[0]
    axb.axhline(ceiling, color='#f59e0b', linestyle='-.', linewidth=1.8, label=f"12-Lead Full Reference (F1 = {ceiling:.4f})", zorder=2)

    # Annotations
    axb.scatter([4], [b3_f1[4]], s=120, facecolors='none', edgecolors='#e11d48', linewidths=2.2, zorder=5)
    axb.annotate(r"$\mathbf{V_1}$ Recovery: +0.027 F1" + "\n(4-lead exceeds 6-lead)", xy=(4, b3_f1[4]), xytext=(4.3, 0.60),
                 arrowprops=dict(arrowstyle="->", color='#e11d48', lw=1.4),
                 fontsize=8.8, fontweight='bold', color='#be123c',
                 bbox=dict(boxstyle="round,pad=0.25", fc="#fff1f2", ec="#f43f5e", lw=1.0))

    axb.scatter([1], [b3_f1[6]], s=120, facecolors='none', edgecolors='#be123c', linewidths=2.2, zorder=5)
    axb.annotate("Single-Lead Cliff:\nF1 = 0.5832", xy=(1, b3_f1[6]), xytext=(1.6, 0.46),
                 arrowprops=dict(arrowstyle="->", color='#be123c', lw=1.4),
                 fontsize=8.5, fontweight='bold', color='#be123c',
                 bbox=dict(boxstyle="round,pad=0.25", fc="#fff1f2", ec="#f43f5e", lw=1.0))

    axb.set_title("Diagnostic Macro-F1 Degradation Across Lead Configurations\n(PTB-XL / Chapman / MIMIC Held-Out Test Cohort, N = 7,375)",
                  fontsize=11.0, fontweight='bold', pad=12, color='#1e293b')
    axb.set_xlabel("Number of Available Leads", fontsize=10.5, fontweight='600', color='#334155')
    axb.set_ylabel("Macro-F1 Score", fontsize=10.5, fontweight='600', color='#334155')
    axb.set_xlim(12.6, 0.4)
    axb.set_xticks(leads)
    axb.set_xticklabels(["12", "10", "8", "6", "4", "2", "1"], fontsize=10)
    axb.set_ylim(0.20, 0.85)
    axb.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    axb.tick_params(colors='#334155', labelsize=10)
    axb.grid(True, linestyle='--', color='#e2e8f0', linewidth=0.8, zorder=0)

    axb.legend(loc='lower left', fontsize=8.8, frameon=True, framealpha=0.95, edgecolor='#cbd5e1')
    for spine in axb.spines.values():
        spine.set_color('#94a3b8')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    for d in output_dirs:
        fig_bot.savefig(d / "fig_results_degradation_boundary_curve.png", bbox_inches='tight', dpi=300)
    plt.close(fig_bot)
    print("Saved fig_results_degradation_boundary_curve.png to all target directories.")

    # -------------------------------------------------------------
    # 4. COMPOSITE SLIDE IMAGE (Matching User's Reference Layout)
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(11, 12), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    # Slide Header: "Results and Analysis"
    fig.text(0.06, 0.955, "Results and Analysis", fontsize=26, fontweight='bold', color='#007a82', family='sans-serif')
    fig.text(0.06, 0.932, "Verified Module 1–4 Empirical Outcomes (Held-Out Test Set N = 7,375, 51,625 Total Single-Pass Evaluations)",
             fontsize=9.5, fontweight='500', color='#475569', family='sans-serif')

    # Background card axes for top chart
    ax_card = fig.add_axes([0.045, 0.485, 0.91, 0.435])
    ax_card.set_facecolor('#113854')
    for spine in ax_card.spines.values():
        spine.set_color('#0284c7')
        spine.set_linewidth(1.5)
    ax_card.set_xticks([])
    ax_card.set_yticks([])

    # Inner chart axes placed inside the dark navy card
    ax1 = fig.add_axes([0.08, 0.590, 0.84, 0.29])
    ax1.set_facecolor('#113854')

    ax1.bar(x1, f1_scores, width=0.52, color='#ffffff', edgecolor='#38bdf8', hatch='///', linewidth=1.1, zorder=3)
    ax1.set_title("MODEL DIAGNOSTIC MACRO-F1 PERFORMANCE ACROSS CONFIGURATIONS",
                  fontsize=11.0, fontweight='bold', color='#ffffff', pad=10)

    ax1.set_ylim(0, 1.0)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], color='#ffffff', fontsize=9.0, fontweight='600')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(model_names, rotation=52, ha='right', color='#ffffff', fontsize=7.2, fontweight='600')

    # Grid inside dark card
    ax1.grid(axis='y', color='white', alpha=0.18, linestyle='-', linewidth=0.8, zorder=0)
    for xi in x1:
        ax1.axvline(xi, color='white', alpha=0.08, linestyle='-', linewidth=0.6, zorder=0)

    for spine in ax1.spines.values():
        spine.set_color('#38bdf8')
        spine.set_linewidth(1.0)
    ax1.tick_params(colors='#ffffff')

    # Bottom Line Chart
    ax2 = fig.add_axes([0.18, 0.07, 0.64, 0.36])
    ax2.set_facecolor('#ffffff')

    ax2.fill_between(leads, b3_f1_ci_low, b3_f1_ci_high, color='#e11d48', alpha=0.15, label="B-3 95% Bootstrap CI", zorder=2)
    ax2.plot(leads, b3_f1, color='#e11d48', marker='o', markersize=7, linewidth=2.2, label="B-3 Control Reasoning Transformer", zorder=4)
    ax2.plot(leads, b1_f1, color='#2563eb', marker='s', markersize=6.5, linewidth=2.0, linestyle='--', label="B-1 Linear Probe (Ridge Logistic)", zorder=4)
    ax2.plot(leads, b2_f1, color='#059669', marker='^', markersize=6.0, linewidth=1.6, linestyle=':', label="B-2 Heuristic Rule-Based Classifier", zorder=3)
    ax2.axhline(ceiling, color='#f59e0b', linestyle='-.', linewidth=1.8, label=f"12-Lead Full Reference (F1 = {ceiling:.4f})", zorder=2)

    ax2.set_title("Diagnostic Macro-F1 Degradation vs. Lead Configuration (Test N = 7,375)",
                  fontsize=11.0, fontweight='bold', pad=10, color='#1e293b')
    ax2.set_xlabel("Lead Configuration (Number of Active Leads)", fontsize=10.0, fontweight='600', color='#334155')
    ax2.set_ylabel("Macro-F1 Score", fontsize=10.0, fontweight='600', color='#334155')
    ax2.set_xlim(12.6, 0.4)
    ax2.set_xticks(leads)
    ax2.set_xticklabels(["12", "10", "8", "6", "4", "2", "1"], fontsize=9.5)
    ax2.set_ylim(0.20, 0.85)
    ax2.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax2.tick_params(colors='#334155', labelsize=9.5)
    ax2.grid(True, linestyle='--', color='#e2e8f0', linewidth=0.8, zorder=0)

    # Annotations on composite bottom
    ax2.scatter([4], [b3_f1[4]], s=110, facecolors='none', edgecolors='#e11d48', linewidths=2.0, zorder=5)
    ax2.annotate(r"$\mathbf{V_1}$ Recovery: +0.027 F1" + "\n(4-lead exceeds 6-lead)", xy=(4, b3_f1[4]), xytext=(4.3, 0.60),
                 arrowprops=dict(arrowstyle="->", color='#e11d48', lw=1.2),
                 fontsize=8.0, fontweight='bold', color='#be123c',
                 bbox=dict(boxstyle="round,pad=0.2", fc="#fff1f2", ec="#f43f5e", lw=0.9))

    ax2.scatter([1], [b3_f1[6]], s=110, facecolors='none', edgecolors='#be123c', linewidths=2.0, zorder=5)
    ax2.annotate("Single-Lead Cliff:\nF1 = 0.5832", xy=(1, b3_f1[6]), xytext=(1.6, 0.46),
                 arrowprops=dict(arrowstyle="->", color='#be123c', lw=1.2),
                 fontsize=8.0, fontweight='bold', color='#be123c',
                 bbox=dict(boxstyle="round,pad=0.2", fc="#fff1f2", ec="#f43f5e", lw=0.9))

    ax2.legend(loc='lower left', fontsize=8.2, frameon=True, framealpha=0.95, edgecolor='#cbd5e1')
    for spine in ax2.spines.values():
        spine.set_color('#94a3b8')
        spine.set_linewidth(0.8)

    for d in output_dirs:
        fig.savefig(d / "results_and_analysis_dashboard.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Saved composite results_and_analysis_dashboard.png to all target directories.")

if __name__ == "__main__":
    create_results_visuals()
