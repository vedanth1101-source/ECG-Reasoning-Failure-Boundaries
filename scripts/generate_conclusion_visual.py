"""
Generate Publication-Quality Clinical White-Paper Infographic
Matching the Conclusion & Future Work Spoken Script.

Three Panels:
1. Established Foundations (Modules 1-4 Data & Reasoning Flow)
2. Core Research Dilemma (Evidence Sufficiency vs. Uncertainty Expression)
3. Next Steps & Future Roadmap (Clinical Knowledge, Grounding Gate, Transparent AI)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch
import numpy as np

def create_conclusion_figure(output_png_path, output_pdf_path):
    # Set high-resolution figure canvas (Widescreen 16:9)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=300, facecolor='#ffffff')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette: Clinical White Paper Theme
    c_bg = '#ffffff'
    c_navy_primary = '#0f172a'
    c_navy_text = '#1e293b'
    c_muted_text = '#64748b'
    
    c_blue = '#2563eb'
    c_blue_light = '#eff6ff'
    c_blue_border = '#bfdbfe'

    c_green = '#059669'
    c_green_light = '#ecfdf5'
    c_green_border = '#a7f3d0'

    c_amber = '#d97706'
    c_amber_light = '#fffbeb'
    c_amber_border = '#fde68a'

    c_red = '#dc2626'
    c_red_light = '#fef2f2'
    c_red_border = '#fecaca'

    c_purple = '#7c3aed'
    c_purple_light = '#f5f3ff'
    c_purple_border = '#ddd6fe'

    c_teal = '#0d9488'
    c_teal_light = '#f0fdfa'
    c_teal_border = '#99f6e4'

    # -------------------------------------------------------------
    # HEADER BANNER
    # -------------------------------------------------------------
    ax.text(50, 96.6, "TOWARDS EVIDENCE-AWARE & TRANSPARENT ECG AI", 
            fontsize=21, weight='bold', color=c_navy_primary, ha='center', va='center', fontfamily='sans-serif')
    ax.text(50, 93.8, "Synthesis of Completed Foundations, Evidence Observability Dilemma, and Clinical Verification Roadmap", 
            fontsize=12, color=c_muted_text, ha='center', va='center', fontfamily='sans-serif')

    # Dividing rule below header
    ax.plot([3.5, 96.5], [92.0, 92.0], color='#e2e8f0', lw=1.5)

    # -------------------------------------------------------------
    # HELPER FUNCTIONS
    # -------------------------------------------------------------
    def draw_card(x, y, w, h, bg_col, border_col, corner_radius=1.5, lw=1.2, zorder=2):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0,rounding_size={corner_radius}",
                             facecolor=bg_col, edgecolor=border_col, linewidth=lw, zorder=zorder)
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, color='#94a3b8', lw=1.5, style='-|>', zorder=3):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle=f'{style},head_width=3.5,head_length=5',
                                color=color, linewidth=lw, zorder=zorder)
        ax.add_patch(arrow)

    # Global Vertical Metrics
    panel_y = 5.5
    panel_h = 84.8
    hdr_h = 6.0
    hdr_y = panel_y + panel_h - hdr_h - 1.2
    bot_y = panel_y + 1.4
    bot_h = 4.2

    # =============================================================
    # PANEL 1: ESTABLISHED FOUNDATIONS (LEFT)
    # =============================================================
    p1_x, p1_w = 3.5, 29.2
    draw_card(p1_x, panel_y, p1_w, panel_h, '#f8fafc', '#e2e8f0', corner_radius=2, lw=1.2)

    # Panel 1 Header
    draw_card(p1_x + 1.2, hdr_y, p1_w - 2.4, hdr_h, c_blue_light, c_blue_border, corner_radius=1.2)
    ax.text(p1_x + p1_w/2, hdr_y + 3.8, "1. ESTABLISHED FOUNDATIONS", 
            fontsize=12.5, weight='bold', color=c_blue, ha='center', va='center')
    ax.text(p1_x + p1_w/2, hdr_y + 1.8, "End-to-End Control Pipeline (Modules 1 - 4)", 
            fontsize=8.5, color='#475569', ha='center', va='center')

    # Flow Cards in Panel 1
    flow_cards = [
        {
            "step": "MODULE 01",
            "title": "Foundational Data Pipeline",
            "desc": "Standardized multi-cohort ingestion (PTB-XL, Chapman,\nMIMIC-IV), bandpass filtering, and 12-lead tensor curation.",
            "tag": "12-Lead Ingestion",
            "tag_bg": c_blue_light, "tag_fg": c_blue, "tag_border": c_blue_border
        },
        {
            "step": "MODULE 02",
            "title": "Lead-Degradation Framework",
            "desc": "Controlled physiological sensor masking simulating real-world\nreduced-lead telemetry without artificial waveform inpainting.",
            "tag": "Physiological Masking",
            "tag_bg": '#f1f5f9', "tag_fg": '#334155', "tag_border": '#cbd5e1'
        },
        {
            "step": "MODULE 03",
            "title": "ECG-FM Representation Pipeline",
            "desc": "Frozen 90.8M parameter foundation model extracting 768-d\nZ_ECG latents paired with rule-extracted clinical observations.",
            "tag": "Z_ECG Latents",
            "tag_bg": c_purple_light, "tag_fg": c_purple, "tag_border": c_purple_border
        },
        {
            "step": "MODULE 04",
            "title": "ECG-Only Control Model",
            "desc": "Baseline clinical reasoning transformer evaluating cross-attention\ntraces and diagnostic predictions from ECG signals alone.",
            "tag": "Control Reasoner",
            "tag_bg": c_green_light, "tag_fg": c_green, "tag_border": c_green_border
        }
    ]

    c_w = p1_w - 2.4
    c_h = 13.5
    gap_y = 3.6
    card_start_y = hdr_y - 2.2 - c_h

    for i, c in enumerate(flow_cards):
        cy = card_start_y - i * (c_h + gap_y)
        cx = p1_x + 1.2
        draw_card(cx, cy, c_w, c_h, '#ffffff', '#cbd5e1', corner_radius=1.2, lw=1)
        
        # Step label & Badge
        ax.text(cx + 1.4, cy + c_h - 2.4, c["step"], fontsize=8.5, weight='bold', color='#64748b')
        
        # Mini Badge
        bw = len(c["tag"]) * 0.45 + 2.5
        draw_card(cx + c_w - bw - 1.4, cy + c_h - 3.4, bw, 2.4, c["tag_bg"], c["tag_border"], corner_radius=0.6, lw=0.8)
        ax.text(cx + c_w - 1.4 - bw/2, cy + c_h - 2.2, c["tag"], fontsize=7.5, weight='bold', color=c["tag_fg"], ha='center', va='center')

        # Title
        ax.text(cx + 1.4, cy + c_h - 5.2, c["title"], fontsize=10.5, weight='bold', color=c_navy_text)
        
        # Description
        ax.text(cx + 1.4, cy + c_h - 7.5, c["desc"], fontsize=8.0, color='#475569',
                va='top', linespacing=1.35)

        # Draw connecting arrow to next card
        if i < len(flow_cards) - 1:
            draw_arrow(p1_x + p1_w/2, cy - 0.2, p1_x + p1_w/2, cy - gap_y + 0.2, color='#94a3b8', lw=1.4)

    # Bottom Banner for Panel 1
    draw_card(p1_x + 1.2, bot_y, c_w, bot_h, '#f8fafc', '#94a3b8', corner_radius=0.8, lw=0.9)
    ax.text(p1_x + p1_w/2, bot_y + bot_h/2, "STATUS: Validated Baseline Architecture Complete", 
            fontsize=8.5, weight='bold', color='#334155', ha='center', va='center')

    # =============================================================
    # PANEL 2: CORE RESEARCH DILEMMA (CENTER)
    # =============================================================
    p2_x, p2_w = 34.6, 30.8
    draw_card(p2_x, panel_y, p2_w, panel_h, '#f8fafc', '#e2e8f0', corner_radius=2, lw=1.2)

    # Panel 2 Header
    draw_card(p2_x + 1.2, hdr_y, p2_w - 2.4, hdr_h, c_amber_light, c_amber_border, corner_radius=1.2)
    ax.text(p2_x + p2_w/2, hdr_y + 3.8, "2. THE CORE RESEARCH PROBLEM", 
            fontsize=12.5, weight='bold', color=c_amber, ha='center', va='center')
    ax.text(p2_x + p2_w/2, hdr_y + 1.8, "Evidence Sufficiency vs. Epistemic Uncertainty", 
            fontsize=8.5, color='#475569', ha='center', va='center')

    # Central Contrast Visualization Card: The Divergence Zone
    div_w = p2_w - 2.4
    div_h = 32.5
    div_y = hdr_y - 2.0 - div_h
    div_x = p2_x + 1.2
    draw_card(div_x, div_y, div_w, div_h, '#ffffff', '#cbd5e1', corner_radius=1.5, lw=1)

    ax.text(div_x + div_w/2, div_y + div_h - 2.6, "THE SENSOR-EVIDENCE INTEGRITY GAP", 
            fontsize=10.5, weight='bold', color=c_navy_primary, ha='center', va='center')
    ax.text(div_x + div_w/2, div_y + div_h - 4.6, "Observed Behavior Under Lead Degradation (12-Lead to Reduced Leads)", 
            fontsize=8, color=c_muted_text, ha='center', va='center')

    # Conceptual Divergence Graph inside the Card
    gx_left, gx_right = div_x + 4.5, div_x + div_w - 4.0
    gy_bottom, gy_top = div_y + 4.8, div_y + div_h - 7.5
    
    # Axes
    ax.plot([gx_left, gx_right], [gy_bottom, gy_bottom], color='#94a3b8', lw=1.2)
    ax.plot([gx_left, gx_left], [gy_bottom, gy_top], color='#94a3b8', lw=1.2)
    ax.text(gx_left - 0.6, gy_top, "Score / Level", fontsize=7.5, color='#64748b', ha='right', va='center')
    ax.text(gx_left, gy_bottom - 2.2, "Full 12 Leads", fontsize=7.5, color='#64748b', ha='center', va='top')
    ax.text(gx_right, gy_bottom - 2.2, "Reduced Leads (e.g. 2-Lead)", fontsize=7.5, color='#64748b', ha='center', va='top')

    # Curve 1: Prediction Confidence (Stays deceptively high)
    cx_vals = np.linspace(gx_left, gx_right, 50)
    cy_conf = gy_top - 1.2 - 1.0 * ((cx_vals - gx_left)/(gx_right - gx_left))**1.8
    ax.plot(cx_vals, cy_conf, color=c_blue, lw=2.4, zorder=4)
    ax.text(gx_left + 1.2, gy_top + 0.3, "Model Diagnostic Confidence (Stays High)", 
            fontsize=8, weight='bold', color=c_blue, ha='left', va='bottom')

    # Curve 2: Physical Sensor Observability (Drops sharply)
    cy_obs = gy_bottom + 1.2 + (gy_top - gy_bottom - 2.6) * (1 - (cx_vals - gx_left)/(gx_right - gx_left))**1.4
    ax.plot(cx_vals, cy_obs, color=c_red, lw=2.4, ls='--', zorder=4)
    ax.text(gx_left + 1.2, gy_bottom + 1.6, "Physical Sensor Observability (Collapses)", 
            fontsize=8, weight='bold', color=c_red, ha='left', va='bottom')

    # Shaded Divergence Danger Area
    ax.fill_between(cx_vals, cy_obs, cy_conf, where=(cy_conf >= cy_obs), color=c_red_light, alpha=0.55, zorder=3)
    
    # Limitation callout box placed cleanly in upper-right divergence space with opaque background
    callout_x, callout_y = (gx_left + gx_right)/2 + 2.2, (gy_bottom + gy_top)/2 + 3.8
    ax.text(callout_x, callout_y, 
            "EVALUATION LIMITATION:\nModels cite unrecorded sensors\nwithout expressing uncertainty.", 
            fontsize=7.4, weight='bold', color=c_red, ha='center', va='center', zorder=6,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor=c_red_border, lw=1.0))

    # The Decision Boundary Split (Sufficient Evidence vs. Uncertainty)
    split_h = 31.0
    split_y = div_y - 2.0 - split_h
    split_w = div_w
    draw_card(div_x, split_y, split_w, split_h, '#ffffff', '#cbd5e1', corner_radius=1.5, lw=1)

    ax.text(div_x + split_w/2, split_y + split_h - 2.6, "DESIRED CLINICAL DECISION BEHAVIOR", 
            fontsize=10.5, weight='bold', color=c_navy_primary, ha='center', va='center')
    ax.text(div_x + split_w/2, split_y + split_h - 4.6, "Guiding Models Between Confident Claims and Honest Abstention", 
            fontsize=8, color=c_muted_text, ha='center', va='center')

    # Sub-card A: Sufficient Evidence -> Predict
    b_w = split_w - 2.4
    b_h = 10.2
    b_x = div_x + 1.2
    b1_y = split_y + split_h - 16.5
    draw_card(b_x, b1_y, b_w, b_h, c_green_light, c_green_border, corner_radius=1.2, lw=1)
    
    # Badge
    draw_card(b_x + 1.2, b1_y + b_h - 4.2, 3.8, 2.8, '#ffffff', c_green_border, corner_radius=0.6)
    ax.text(b_x + 3.1, b1_y + b_h - 2.8, "REGIME A", fontsize=6.8, weight='bold', color=c_green, ha='center', va='center')
    ax.text(b_x + 5.8, b1_y + b_h - 2.8, "Sufficient Evidence -> Support Prediction", fontsize=9.2, weight='bold', color=c_green)
    ax.text(b_x + 5.8, b1_y + b_h - 5.5, "When required anatomical leads are active & verified:\nSystem produces a confident, well-grounded diagnostic prediction.",
            fontsize=7.6, color='#065f46', linespacing=1.3)

    # Sub-card B: Evidence Degraded -> Express Uncertainty
    b2_y = split_y + 2.2
    draw_card(b_x, b2_y, b_w, b_h, c_amber_light, c_amber_border, corner_radius=1.2, lw=1)
    
    draw_card(b_x + 1.2, b2_y + b_h - 4.2, 3.8, 2.8, '#ffffff', c_amber_border, corner_radius=0.6)
    ax.text(b_x + 3.1, b2_y + b_h - 2.8, "REGIME B", fontsize=6.8, weight='bold', color=c_amber, ha='center', va='center')
    ax.text(b_x + 5.8, b2_y + b_h - 2.8, "Evidence Degraded -> Express Uncertainty", fontsize=9.2, weight='bold', color=c_amber)
    ax.text(b_x + 5.8, b2_y + b_h - 5.5, "When critical leads are missing & vectors unobservable:\nSystem abstains from hallucinating & signals epistemic uncertainty.",
            fontsize=7.6, color='#92400e', linespacing=1.3)

    # Bottom Banner for Panel 2
    draw_card(p2_x + 1.2, bot_y, p2_w - 2.4, bot_h, '#fef3c7', '#f59e0b', corner_radius=0.8, lw=0.9)
    ax.text(p2_x + p2_w/2, bot_y + bot_h/2, "CORE MISSION: Distinguish Grounded Insight from Overconfident Guesswork", 
            fontsize=8.5, weight='bold', color='#92400e', ha='center', va='center')

    # =============================================================
    # PANEL 3: NEXT STEPS & FUTURE ROADMAP (RIGHT)
    # =============================================================
    p3_x, p3_w = 67.3, 29.2
    draw_card(p3_x, panel_y, p3_w, panel_h, '#f8fafc', '#e2e8f0', corner_radius=2, lw=1.2)

    # Panel 3 Header
    draw_card(p3_x + 1.2, hdr_y, p3_w - 2.4, hdr_h, c_purple_light, c_purple_border, corner_radius=1.2)
    ax.text(p3_x + p3_w/2, hdr_y + 3.8, "3. NEXT STEPS & FUTURE ROADMAP", 
            fontsize=12.5, weight='bold', color=c_purple, ha='center', va='center')
    ax.text(p3_x + p3_w/2, hdr_y + 1.8, "Towards Transparent, Evidence-Aware AI", 
            fontsize=8.5, color='#475569', ha='center', va='center')

    # 3 Next Steps Cards
    next_steps = [
        {
            "num": "NEXT STEP 01",
            "title": "Clinical Knowledge-Assisted Path",
            "desc": "Incorporate structured electrophysiological priors (SNOMED-CT relations)\nto reroute attention toward observable alternate leads during degradation.",
            "status": "In Development",
            "badge_bg": c_purple_light, "badge_fg": c_purple, "badge_bd": c_purple_border,
            "feat": "Architecture: SNOMED-CT Priors + Frontal Re-routing"
        },
        {
            "num": "NEXT STEP 02",
            "title": "Evidence-Grounding Checks",
            "desc": "Deploy deterministic ternary verification gate classifying claims as\nSUPPORTED, CONTRADICTED, or UNAVAILABLE against active hardware masks.",
            "status": "Gate Design Ready",
            "badge_bg": c_teal_light, "badge_fg": c_teal, "badge_bd": c_teal_border,
            "feat": "Ternary Gate: [SUPPORTED] [CONTRADICTED] [UNAVAILABLE]"
        },
        {
            "num": "NEXT STEP 03",
            "title": "Controlled Protocol Comparison",
            "desc": "Execute multi-cohort comparative benchmarking under formal clinical\nprotocols to quantify safety boundaries, RFB thresholds, and trade-offs.",
            "status": "Protocol Specified",
            "badge_bg": c_blue_light, "badge_fg": c_blue, "badge_bd": c_blue_border,
            "feat": "Audit Protocol: Reasoning Failure Boundary (RFB) Benchmark"
        }
    ]

    s_w = p3_w - 2.4
    s_h = 18.0
    s_gap = 4.4
    step_start_y = hdr_y - 2.2 - s_h

    for i, s in enumerate(next_steps):
        sy = step_start_y - i * (s_h + s_gap)
        sx = p3_x + 1.2
        draw_card(sx, sy, s_w, s_h, '#ffffff', '#cbd5e1', corner_radius=1.2, lw=1)

        # Label & Badge
        ax.text(sx + 1.4, sy + s_h - 2.4, s["num"], fontsize=8.5, weight='bold', color='#64748b')
        
        bw = len(s["status"]) * 0.45 + 2.5
        draw_card(sx + s_w - bw - 1.4, sy + s_h - 3.4, bw, 2.4, s["badge_bg"], s["badge_bd"], corner_radius=0.6, lw=0.8)
        ax.text(sx + s_w - 1.4 - bw/2, sy + s_h - 2.2, s["status"], fontsize=7.5, weight='bold', color=s["badge_fg"], ha='center', va='center')

        # Title
        ax.text(sx + 1.4, sy + s_h - 5.2, s["title"], fontsize=10.5, weight='bold', color=c_navy_text)
        
        # Description
        ax.text(sx + 1.4, sy + s_h - 7.5, s["desc"], fontsize=8.0, color='#475569',
                va='top', linespacing=1.35)

        # Mini visual feature tag
        draw_card(sx + 1.4, sy + 1.6, s_w - 2.8, 3.4, '#f8fafc', '#e2e8f0', corner_radius=0.6, lw=0.8)
        ax.text(sx + 2.5, sy + 3.3, s["feat"], fontsize=7.6, weight='bold', color='#334155', va='center')

        # Arrow down
        if i < len(next_steps) - 1:
            draw_arrow(p3_x + p3_w/2, sy - 0.2, p3_x + p3_w/2, sy - s_gap + 0.2, color='#94a3b8', lw=1.4)

    # Bottom Banner for Panel 3: Ultimate Mission
    draw_card(p3_x + 1.2, bot_y, s_w, bot_h, '#ecfdf5', c_green_border, corner_radius=0.8, lw=0.9)
    ax.text(p3_x + p3_w/2, bot_y + bot_h/2, "ULTIMATE GOAL: Transparent & Evidence-Aware Medical AI", 
            fontsize=8.5, weight='bold', color='#065f46', ha='center', va='center')

    # -------------------------------------------------------------
    # FOOTER CITATION / CAPTION
    # -------------------------------------------------------------
    ax.text(50, 2.2, 
            "Figure: Conceptual Framework Synthesizing Established Foundations, Evidence Observability Gap, and Transparent Evaluation Roadmap | Batch 43 (SSN CSE)",
            fontsize=9, color='#64748b', ha='center', va='center', style='italic')

    # Save outputs
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    plt.savefig(output_png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(output_pdf_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"Generated 3-Panel PNG: {output_png_path}")
    print(f"Generated 3-Panel PDF: {output_pdf_path}")

    # Also crop and save individual panels for modular presentation use
    from PIL import Image
    base_img = Image.open(output_png_path)
    w, h = base_img.size

    # Panel 1 Crop (Left: Foundations)
    p1_crop = base_img.crop((int(0.02 * w), int(0.04 * h), int(0.345 * w), int(0.96 * h)))
    p1_path = os.path.join(os.path.dirname(output_png_path), "conclusion_panel_1_foundations.png")
    p1_crop.save(p1_path)

    # Panel 2 Crop (Center: Evidence vs Uncertainty Dilemma)
    p2_crop = base_img.crop((int(0.340 * w), int(0.04 * h), int(0.665 * w), int(0.96 * h)))
    p2_path = os.path.join(os.path.dirname(output_png_path), "conclusion_panel_2_evidence_dilemma.png")
    p2_crop.save(p2_path)

    # Panel 3 Crop (Right: Next Steps & Roadmap)
    p3_crop = base_img.crop((int(0.660 * w), int(0.04 * h), int(0.980 * w), int(0.96 * h)))
    p3_path = os.path.join(os.path.dirname(output_png_path), "conclusion_panel_3_future_roadmap.png")
    p3_crop.save(p3_path)

    print(f"Generated Panel 1: {p1_path}")
    print(f"Generated Panel 2: {p2_path}")
    print(f"Generated Panel 3: {p3_path}")

if __name__ == '__main__':
    png_path = "e:/ECG/visualizations/conclusion_evidence_roadmap.png"
    pdf_path = "e:/ECG/visualizations/conclusion_evidence_roadmap.pdf"
    create_conclusion_figure(png_path, pdf_path)
