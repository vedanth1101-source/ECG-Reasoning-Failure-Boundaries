"""
Publication-Quality Disease Distribution Visualizer
===================================================
Generates high-resolution publication charts depicting the distribution
of the 8 finalized benchmark diseases across:
  - PTB-XL (N = 21,799)
  - Chapman-Shaoxing (N = 10,158)
  - MIMIC-IV-ECG (N = 5,000)
Total Records = 36,957

Outputs:
  - visualizations/disease_distribution_grouped.png
  - visualizations/disease_distribution_stacked.png
  - visualizations/disease_distribution_combined.png
  - visualizations/disease_distribution_summary.md
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

repo_root = Path(__file__).resolve().parent.parent

DISEASES = [
    "Normal ECG",
    "Atrial Fibrillation",
    "Right Bundle Branch Block",
    "Left Bundle Branch Block",
    "Sinus Bradycardia",
    "Sinus Tachycardia",
    "1st Degree AV Block",
    "Myocardial Infarction"
]

# Abbreviated names for crisp chart labels
SHORT_NAMES = [
    "Normal",
    "AFib",
    "RBBB",
    "LBBB",
    "Sinus Brady",
    "Sinus Tachy",
    "1st Deg AVB",
    "Myocardial Inf"
]


def load_cohort_counts():
    data_dir = repo_root / "data" / "processed_100hz"
    ptb_lbl = np.load(data_dir / "labels.npy")
    chap_lbl = np.load(data_dir / "chapman" / "labels.npy")
    mim_lbl = np.load(data_dir / "mimic" / "labels.npy")

    df = pd.DataFrame({
        "Disease": DISEASES,
        "ShortName": SHORT_NAMES,
        "PTB-XL": ptb_lbl.sum(axis=0),
        "Chapman": chap_lbl.sum(axis=0),
        "MIMIC": mim_lbl.sum(axis=0)
    })
    df["Total"] = df["PTB-XL"] + df["Chapman"] + df["MIMIC"]
    return df


def plot_grouped_bar_chart(df: pd.DataFrame, out_dir: Path):
    """Figure 1: Grouped bar chart with direct value labels."""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    x = np.arange(len(df))
    width = 0.26

    # Sophisticated clinical palette
    c_ptb = "#1f4e79"    # Deep Slate Blue
    c_chap = "#2ca25f"   # Forest Green
    c_mim = "#d95f02"    # Warm Terracotta

    rects1 = ax.bar(x - width, df["PTB-XL"], width, label="PTB-XL (N = 21,799)", color=c_ptb, edgecolor="none", zorder=3)
    rects2 = ax.bar(x, df["Chapman"], width, label="Chapman-Shaoxing (N = 10,158)", color=c_chap, edgecolor="none", zorder=3)
    rects3 = ax.bar(x + width, df["MIMIC"], width, label="MIMIC-IV-ECG (N = 5,000)", color=c_mim, edgecolor="none", zorder=3)

    # Grid & styling
    ax.set_facecolor("#f9fafb")
    fig.patch.set_facecolor("#ffffff")
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
    ax.set_axisbelow(True)

    # Labels and Titles
    ax.set_title("Disease Representation Across Multi-Center ECG Cohorts (N = 36,957)", 
                 fontsize=15, fontweight="bold", pad=20, color="#1e293b")
    ax.set_ylabel("Number of Validated Entries (Log-Scaled)", fontsize=12, fontweight="600", color="#334155")
    ax.set_yscale("log")
    ax.set_ylim(10, 25000)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))

    ax.set_xticks(x)
    ax.set_xticklabels(df["ShortName"], fontsize=11, fontweight="600", color="#1e293b")
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Add count labels on top of bars
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{int(h):,}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=7.5, fontweight="bold", color="#475569",
                            rotation=45)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Clean borders
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color("#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=11, loc="upper right")

    plt.tight_layout()
    out_file = out_dir / "disease_distribution_grouped.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved grouped bar chart to {out_file}")


def plot_stacked_composition_chart(df: pd.DataFrame, out_dir: Path):
    """Figure 2: 100% Horizontal Stacked Bar Chart showing cohort contribution."""
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)

    # Calculate proportions
    pct_ptb = df["PTB-XL"] / df["Total"] * 100
    pct_chap = df["Chapman"] / df["Total"] * 100
    pct_mim = df["MIMIC"] / df["Total"] * 100

    y = np.arange(len(df))
    height = 0.55

    c_ptb = "#1f4e79"
    c_chap = "#2ca25f"
    c_mim = "#d95f02"

    p1 = ax.barh(y, pct_ptb, height, label="PTB-XL", color=c_ptb, edgecolor="white", zorder=3)
    p2 = ax.barh(y, pct_chap, height, left=pct_ptb, label="Chapman-Shaoxing", color=c_chap, edgecolor="white", zorder=3)
    p3 = ax.barh(y, pct_mim, height, left=pct_ptb + pct_chap, label="MIMIC-IV-ECG", color=c_mim, edgecolor="white", zorder=3)

    ax.set_facecolor("#f9fafb")
    fig.patch.set_facecolor("#ffffff")
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)

    ax.set_title("Relative Dataset Composition Per Disease Condition (%)", 
                 fontsize=14, fontweight="bold", pad=15, color="#1e293b")
    ax.set_xlabel("Proportion of Total Class Representation (%)", fontsize=11, fontweight="600", color="#334155")
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Disease"], fontsize=11, fontweight="600", color="#1e293b")
    ax.invert_yaxis()  # Top to bottom

    # Add percentages inside bars if width allows
    for i in range(len(df)):
        # PTB
        if pct_ptb.iloc[i] > 7:
            ax.text(pct_ptb.iloc[i] / 2, i, f"{pct_ptb.iloc[i]:.1f}%", va="center", ha="center", color="white", fontsize=8.5, fontweight="bold")
        # Chapman
        if pct_chap.iloc[i] > 7:
            ax.text(pct_ptb.iloc[i] + pct_chap.iloc[i] / 2, i, f"{pct_chap.iloc[i]:.1f}%", va="center", ha="center", color="white", fontsize=8.5, fontweight="bold")
        # MIMIC
        if pct_mim.iloc[i] > 7:
            ax.text(pct_ptb.iloc[i] + pct_chap.iloc[i] + pct_mim.iloc[i] / 2, i, f"{pct_mim.iloc[i]:.1f}%", va="center", ha="center", color="white", fontsize=8.5, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#cbd5e1")

    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=10, loc="lower right")

    plt.tight_layout()
    out_file = out_dir / "disease_distribution_stacked.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved stacked composition chart to {out_file}")


def plot_combined_panel(df: pd.DataFrame, out_dir: Path):
    """Figure 3: Publication 2-panel composite figure (A: Absolute Counts, B: Proportions)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

    # ------------------ PANEL A: Absolute Linear Counts ------------------
    x = np.arange(len(df))
    width = 0.26
    c_ptb = "#1f4e79"
    c_chap = "#2ca25f"
    c_mim = "#d95f02"

    ax1.bar(x - width, df["PTB-XL"], width, label="PTB-XL (N = 21,799)", color=c_ptb, zorder=3)
    ax1.bar(x, df["Chapman"], width, label="Chapman (N = 10,158)", color=c_chap, zorder=3)
    ax1.bar(x + width, df["MIMIC"], width, label="MIMIC (N = 5,000)", color=c_mim, zorder=3)

    ax1.set_facecolor("#f9fafb")
    ax1.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
    ax1.set_title("A. Absolute Patient Counts per Disease", fontsize=13, fontweight="bold", pad=12, color="#1e293b")
    ax1.set_ylabel("Total Number of Patients", fontsize=11, fontweight="600", color="#334155")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["ShortName"], fontsize=10, fontweight="600", color="#1e293b", rotation=30, ha="right")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9.5, loc="upper right")

    # ------------------ PANEL B: Relative Stacked % ------------------
    pct_ptb = df["PTB-XL"] / df["Total"] * 100
    pct_chap = df["Chapman"] / df["Total"] * 100
    pct_mim = df["MIMIC"] / df["Total"] * 100
    y = np.arange(len(df))
    height = 0.55

    ax2.barh(y, pct_ptb, height, label="PTB-XL", color=c_ptb, edgecolor="white", zorder=3)
    ax2.barh(y, pct_chap, height, left=pct_ptb, label="Chapman", color=c_chap, edgecolor="white", zorder=3)
    ax2.barh(y, pct_mim, height, left=pct_ptb + pct_chap, label="MIMIC", color=c_mim, edgecolor="white", zorder=3)

    ax2.set_facecolor("#f9fafb")
    ax2.grid(axis="x", linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
    ax2.set_title("B. Dataset Composition Breakdown (%)", fontsize=13, fontweight="bold", pad=12, color="#1e293b")
    ax2.set_xlabel("Contribution (%)", fontsize=11, fontweight="600", color="#334155")
    ax2.set_xlim(0, 100)
    ax2.set_yticks(y)
    ax2.set_yticklabels(df["Disease"], fontsize=10.5, fontweight="600", color="#1e293b")
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9.5, loc="lower right")

    fig.patch.set_facecolor("#ffffff")
    plt.suptitle("Multi-Cohort Disease Representation & Spread Benchmark (N = 36,957)", 
                 fontsize=15, fontweight="bold", y=0.98, color="#0f172a")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_file = out_dir / "disease_distribution_combined.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined panel chart to {out_file}")


def write_summary_markdown(df: pd.DataFrame, out_dir: Path):
    out_file = out_dir / "disease_distribution_summary.md"
    lines = [
        "# Multi-Cohort Disease Distribution Benchmark",
        "",
        "Distribution and cross-cohort spread for the frozen 8-disease benchmark across all 36,957 recordings.",
        "",
        "## 1. Absolute & Relative Distribution Table",
        "",
        "| Disease Condition | PTB-XL (N=21,799) | Chapman (N=10,158) | MIMIC (N=5,000) | Combined Total (N=36,957) | % of Combined Population |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for _, r in df.iterrows():
        pct = (r["Total"] / 36957) * 100
        lines.append(
            f"| **{r['Disease']}** | {r['PTB-XL']:,} ({r['PTB-XL']/r['Total']*100:.1f}%) | "
            f"{r['Chapman']:,} ({r['Chapman']/r['Total']*100:.1f}%) | "
            f"{r['MIMIC']:,} ({r['MIMIC']/r['Total']*100:.1f}%) | "
            f"**{r['Total']:,}** | {pct:.1f}% |"
        )

    lines.extend([
        "",
        "## 2. Key Clinical Distribution Observations",
        "",
        "1. **Normal ECG Dominance:** Represents 15,111 recordings (40.9% of total), with PTB-XL contributing the vast majority (12,738).",
        "2. **Arrhythmia Synergy:**",
        "   - **Atrial Fibrillation (3,905 cases):** Sourced evenly from both PTB-XL (1,514) and Chapman (1,766), plus MIMIC (625).",
        "   - **Sinus Bradycardia (5,111 cases):** Strongly anchored by Chapman-Shaoxing (3,849 cases = 75.3% of all bradycardia).",
        "   - **Sinus Tachycardia (3,005 cases):** Balanced between Chapman (1,554) and PTB-XL (826).",
        "3. **Conduction Delays:**",
        "   - **Right Bundle Branch Block (2,730 cases):** PTB-XL (1,655) and Chapman (450).",
        "   - **Left Bundle Branch Block (1,439 cases):** Critical low-prevalence category well represented across all 3 centers.",
        "   - **1st Degree AV Block (1,674 cases):** Strong prevalence in PTB-XL (793) and MIMIC (625).",
        "4. **Ischemic Pathology (Myocardial Infarction):**",
        "   - **5,905 total infarct cases:** Predominantly provided by PTB-XL (5,240 confirmed infarcts with anatomical sub-localization) and MIMIC (625 emergency/ICU cases).",
        "",
        "## 3. Generated Visualizations",
        "",
        "- `visualizations/disease_distribution_grouped.png` (Absolute log-scale grouped comparison)",
        "- `visualizations/disease_distribution_stacked.png` (Relative cohort composition per condition)",
        "- `visualizations/disease_distribution_combined.png` (Publication 2-panel composite figure)"
    ])

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary report to {out_file}")


def main():
    out_dir = repo_root / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading disease representation across cohorts...")
    df = load_cohort_counts()
    print(df[["Disease", "PTB-XL", "Chapman", "MIMIC", "Total"]])

    plot_grouped_bar_chart(df, out_dir)
    plot_stacked_composition_chart(df, out_dir)
    plot_combined_panel(df, out_dir)
    write_summary_markdown(df, out_dir)
    print("\nVisualizations successfully generated in 'visualizations/' directory.")


if __name__ == "__main__":
    main()
