"""
generate_report_figures.py
--------------------------
Reproducible generation of:
  - figures/fig4_1_macro_metrics_by_configuration.png
  - figures/fig4_2_per_condition_f1.png

Draws directly from:
  - runs/module4_v3/metrics/test_evaluation_summary.json
  - runs/module4_v3/metrics/comprehensive_baseline_comparison.json

Outputs saved to:
  - figures/
  - report/figures/
  - report/first_half_implementation/figures/
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def generate_figures():
    root = Path(__file__).resolve().parent.parent
    test_summary_path = root / "runs" / "module4_v3" / "metrics" / "test_evaluation_summary.json"
    baseline_path = root / "runs" / "module4_v3" / "metrics" / "comprehensive_baseline_comparison.json"
    
    with open(test_summary_path, "r", encoding="utf-8") as f:
        test_summary = json.load(f)
        
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_summary = json.load(f)
        
    output_dirs = [
        root / "figures",
        root / "report" / "figures",
        root / "report" / "first_half_implementation" / "figures"
    ]
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    configs = ["12-lead", "10-lead", "8-lead", "6-lead", "4-lead", "2-lead", "1-lead"]
    leads = [12, 10, 8, 6, 4, 2, 1]
    
    # B-3 metrics
    b3_auroc = []
    b3_auroc_ci_low = []
    b3_auroc_ci_high = []
    b3_f1 = []
    b3_f1_ci_low = []
    b3_f1_ci_high = []
    
    # Per-condition F1
    conditions = ["NORM", "AFIB", "RBBB", "LBBB", "SBRAD", "STACH", "1AVB", "MI"]
    cond_f1 = {c: [] for c in conditions}
    
    for cfg in configs:
        data = test_summary["degradation_trajectory"][cfg]
        b3_auroc.append(data["macro_auroc"])
        ci_auroc = data["confidence_intervals"]["macro_auroc_95ci"]
        b3_auroc_ci_low.append(data["macro_auroc"] - ci_auroc[0])
        b3_auroc_ci_high.append(ci_auroc[1] - data["macro_auroc"])
        
        b3_f1.append(data["macro_f1"])
        ci_f1 = data["confidence_intervals"]["macro_f1_95ci"]
        b3_f1_ci_low.append(data["macro_f1"] - ci_f1[0])
        b3_f1_ci_high.append(ci_f1[1] - data["macro_f1"])
        
        for c in conditions:
            cond_f1[c].append(data["per_class_f1"][c])
            
    # B-1 metrics from Table 4.9 / comprehensive baseline
    b1_auroc = [0.9224, 0.9207, 0.9207, 0.8448, 0.8838, 0.8550, 0.7624]
    b1_f1 = [0.6269, 0.6216, 0.6199, 0.4172, 0.4991, 0.4296, 0.2874]
    
    # --- Figure 4.1 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    fig.suptitle("Test partition, seed 42, authorisation-unverified run (descriptive)", fontsize=11, y=0.98)
    
    x = np.arange(len(leads))
    
    # (a) Macro-AUROC
    ax1.errorbar(x, b3_auroc, yerr=[b3_auroc_ci_low, b3_auroc_ci_high], fmt='-o', color='#1f77b4',
                 capsize=4, lw=1.8, ms=6, label="B-3 Control Reasoning Transformer")
    ax1.plot(x, b1_auroc, '--s', color='#ff7f0e', lw=1.8, ms=6, label="B-1 linear probe (no CI available)")
    ax1.set_title("(a) Macro-AUROC", fontsize=14, pad=10)
    ax1.set_xlabel("Lead configuration (number of active leads; not nested)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(leads, fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(True, linestyle='-', alpha=0.3)
    ax1.legend(loc='lower left', fontsize=9)
    
    # (b) Macro-F1
    ax2.errorbar(x, b3_f1, yerr=[b3_f1_ci_low, b3_f1_ci_high], fmt='-o', color='#1f77b4',
                 capsize=4, lw=1.8, ms=6, label="B-3 Control Reasoning Transformer")
    ax2.plot(x, b1_f1, '--s', color='#ff7f0e', lw=1.8, ms=6, label="B-1 linear probe (no CI available)")
    ax2.set_title("(b) Macro-F1", fontsize=14, pad=10)
    ax2.set_xlabel("Lead configuration (number of active leads; not nested)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(leads, fontsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.grid(True, linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    for d in output_dirs:
        fig.savefig(d / "fig4_1_macro_metrics_by_configuration.png", bbox_inches='tight')
    plt.close(fig)
    print("Saved fig4_1_macro_metrics_by_configuration.png to all target directories.")
    
    # --- Figure 4.2 ---
    fig2, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.set_title("B-3, test partition, seed 42 (descriptive)", fontsize=10, pad=8)
    
    colors = {
        "NORM": "#1f77b4",
        "AFIB": "#ff7f0e",
        "RBBB": "#2ca02c",
        "LBBB": "#d62728",
        "SBRAD": "#9467bd",
        "STACH": "#8c564b",
        "1AVB": "#e377c2",
        "MI": "#7f7f7f"
    }
    
    for c in conditions:
        ax.plot(x, cond_f1[c], '-o', color=colors[c], label=c, lw=1.8, ms=6)
        
    ax.set_xlabel("Lead configuration (number of active leads; not nested)", fontsize=11)
    ax.set_ylabel("F1 at threshold 0.5 (calibrated)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(leads, fontsize=12)
    ax.set_ylim([0.0, 1.0])
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.legend(loc='lower left', ncol=4, fontsize=9)
    
    plt.tight_layout()
    for d in output_dirs:
        fig2.savefig(d / "fig4_2_per_condition_f1.png", bbox_inches='tight')
    plt.close(fig2)
    print("Saved fig4_2_per_condition_f1.png to all target directories.")

if __name__ == "__main__":
    generate_figures()
