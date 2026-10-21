"""
Label Harmonization & Disease Candidate Analysis Pipeline
=========================================================
Steps 2, 3, 4, 5 of Dataset Preparation:
1. Extracts raw labels across PTB-XL, Chapman-Shaoxing, and MIMIC-IV-ECG.
2. Generates separate traceable raw label inventories.
3. Implements 3-tier mapping (Raw -> Normalized Concept -> Common Disease).
4. Generates master harmonization mapping catalog.
5. Calculates disease record & patient counts across all datasets.
6. Evaluates predefined eligibility criteria.
7. Proposes the frozen 6-7 disease set.
8. Writes reports:
   - `reports/raw_label_summary.md`
   - `reports/harmonization_table.md`
   - `reports/disease_candidate_analysis.md`
"""

import glob
import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, ".")
from labels.raw_label_extraction import (
    extract_ptbxl_raw_labels,
    extract_chapman_raw_labels
)
from labels.harmonization import (
    harmonize_labels_dataframe,
    generate_harmonization_mapping_table,
    harmonize_mimic_label
)
from labels.disease_analysis import (
    analyze_disease_candidates,
    select_proposed_disease_set
)


def extract_mimic_from_available_parts(mimic_dir: Path) -> pd.DataFrame:
    """
    Extracts raw report labels from assembled machine_measurements.csv
    or downloaded interim part files.
    """
    csv_file = mimic_dir / "machine_measurements.csv"
    if csv_file.exists() and csv_file.stat().st_size > 1000000:
        from labels.raw_label_extraction import extract_mimic_raw_labels
        return extract_mimic_raw_labels(csv_file)
    
    # Otherwise read from downloaded part files
    parts = sorted(list(mimic_dir.glob("part_*.tmp")))
    if not parts:
        raise FileNotFoundError("No MIMIC-IV-ECG measurements files found.")
        
    print(f"[MIMIC-IV-ECG] Reading from {len(parts)} downloaded interim chunks...")
    records = []
    seen_studies = set()
    
    for p in parts:
        try:
            with open(p, "rb") as f:
                lines = f.read().decode("utf-8", errors="ignore").splitlines()
            for line in lines:
                parts_l = line.split(",")
                if len(parts_l) > 4 and parts_l[0].isdigit():
                    sub_id = parts_l[0]
                    study_id = parts_l[1]
                    ecg_time = parts_l[3] if len(parts_l) > 3 else ""
                    
                    # Deduplicate across chunk boundaries
                    study_key = f"{sub_id}_{study_id}"
                    if study_key in seen_studies:
                        continue
                    seen_studies.add(study_key)
                    
                    # Columns 4 to 21 are report_0 to report_17
                    for idx, r_text in enumerate(parts_l[4:22]):
                        r_clean = r_text.strip().strip('"')
                        if r_clean and r_clean.lower() != "nan":
                            records.append({
                                "dataset": "MIMIC-IV-ECG",
                                "record_id": study_id,
                                "patient_id": sub_id,
                                "raw_label": r_clean,
                                "raw_label_code": f"report_{idx}",
                                "report_index": f"report_{idx}",
                                "ecg_time": ecg_time
                            })
        except Exception as e:
            print(f"Error reading {p}: {e}")
            
    df = pd.DataFrame(records)
    print(f"[MIMIC-IV-ECG] Extracted {len(df)} label instances from {df['record_id'].nunique()} records and {df['patient_id'].nunique()} unique patients.")
    return df


def run_pipeline():
    print("==================================================")
    print("STARTING DATASET INSPECTION & LABEL HARMONIZATION")
    print("==================================================")
    
    # -------------------------------------------------------------
    # 1. Raw Label Extraction
    # -------------------------------------------------------------
    print("\n[Step 2] Extracting Raw Labels...")
    
    # PTB-XL
    ptbxl_raw = extract_ptbxl_raw_labels(
        "data/ptbxl/ptbxl_database.csv",
        "labels/mappings/ptbxl_scp_mapping.csv"
    )
    
    # Chapman
    chapman_raw = extract_chapman_raw_labels(
        "data/chapman",
        "labels/mappings/chapman_snomed_mapping.csv"
    )
    
    # MIMIC
    mimic_raw = extract_mimic_from_available_parts(Path("data/mimic"))
    
    # Save raw label summaries
    Path("reports").mkdir(parents=True, exist_ok=True)
    raw_summary_md = f"""# Raw Diagnostic Label Inventory

This report documents original raw diagnostic statements extracted directly from each source dataset without prior normalization.

## Summary

| Dataset | Total Records | Total Unique Patients | Total Raw Label Instances | Unique Diagnostic Terms/Codes |
| :--- | :--- | :--- | :--- | :--- |
| **PTB-XL** | {ptbxl_raw['record_id'].nunique():,} | {ptbxl_raw['patient_id'].nunique():,} | {len(ptbxl_raw):,} | {ptbxl_raw['raw_label_code'].nunique():,} |
| **Chapman-Shaoxing** | {chapman_raw['record_id'].nunique():,} | {chapman_raw['patient_id'].nunique():,} | {len(chapman_raw):,} | {chapman_raw['raw_label_code'].nunique():,} |
| **MIMIC-IV-ECG** | {mimic_raw['record_id'].nunique():,} | {mimic_raw['patient_id'].nunique():,} | {len(mimic_raw):,} | {mimic_raw['raw_label'].nunique():,} |

---

## Top Raw Diagnostic Labels by Dataset

### PTB-XL Top 15 Raw Statements (SCP Codes)
| SCP Code | Description | Instances | Unique Patients |
| :--- | :--- | :--- | :--- |
"""
    ptb_top = ptbxl_raw.groupby(["raw_label_code", "raw_label"]).agg(
        instances=("record_id", "count"),
        patients=("patient_id", "nunique")
    ).sort_values("instances", ascending=False).head(15).reset_index()
    for _, r in ptb_top.iterrows():
        raw_summary_md += f"| `{r['raw_label_code']}` | {r['raw_label']} | {r['instances']:,} | {r['patients']:,} |\n"

    raw_summary_md += """
### Chapman-Shaoxing Top 15 Raw Labels (SNOMED CT)
| SNOMED Code | Diagnostic Label | Instances | Unique Patients |
| :--- | :--- | :--- | :--- |
"""
    chap_top = chapman_raw.groupby(["raw_label_code", "raw_label"]).agg(
        instances=("record_id", "count"),
        patients=("patient_id", "nunique")
    ).sort_values("instances", ascending=False).head(15).reset_index()
    for _, r in chap_top.iterrows():
        raw_summary_md += f"| `{r['raw_label_code']}` | {r['raw_label']} | {r['instances']:,} | {r['patients']:,} |\n"

    raw_summary_md += """
### MIMIC-IV-ECG Top 15 Raw Report Phrases
| Report Phrase | Instances | Unique Patients |
| :--- | :--- | :--- |
"""
    mimic_top = mimic_raw.groupby("raw_label").agg(
        instances=("record_id", "count"),
        patients=("patient_id", "nunique")
    ).sort_values("instances", ascending=False).head(15).reset_index()
    for _, r in mimic_top.iterrows():
        raw_summary_md += f"| {r['raw_label']} | {r['instances']:,} | {r['patients']:,} |\n"

    Path("reports/raw_label_summary.md").write_text(raw_summary_md, encoding="utf-8")
    print("Written reports/raw_label_summary.md")
    
    # -------------------------------------------------------------
    # 2. Harmonization Mapping Table
    # -------------------------------------------------------------
    print("\n[Step 3] Generating Master Harmonization Mapping Table...")
    master_mapping = generate_harmonization_mapping_table()
    master_mapping.to_csv("labels/mappings/harmonization_mapping.csv", index=False)
    
    harm_md = """# Clinical Label Harmonization Mapping Catalog

3-tier mapping pipeline:
$$\\text{RAW DATASET LABEL} \\longrightarrow \\text{NORMALIZED CLINICAL CONCEPT} \\longrightarrow \\text{COMMON DISEASE LABEL}$$

| Dataset | Raw Label / Code | Normalized Clinical Concept | Common Disease Label | Mapping Confidence | Clinical / Physiological Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, r in master_mapping.iterrows():
        harm_md += f"| **{r['dataset']}** | `{r['raw_label']}` | {r['normalized_concept']} | **{r['common_disease']}** | {r['mapping_confidence']} | {r['mapping_notes']} |\n"

    Path("reports/harmonization_table.md").write_text(harm_md, encoding="utf-8")
    print("Written reports/harmonization_table.md")

    # -------------------------------------------------------------
    # 3. Apply Harmonization to Datasets
    # -------------------------------------------------------------
    print("\n[Step 4] Applying Harmonization across datasets...")
    ptbxl_harm = harmonize_labels_dataframe(ptbxl_raw)
    chapman_harm = harmonize_labels_dataframe(chapman_raw)
    mimic_harm = harmonize_labels_dataframe(mimic_raw)
    
    harmonized_dict = {
        "PTB-XL": ptbxl_harm,
        "Chapman-Shaoxing": chapman_harm,
        "MIMIC-IV-ECG": mimic_harm
    }
    
    # -------------------------------------------------------------
    # 4. Disease Candidate Analysis & Eligibility
    # -------------------------------------------------------------
    print("\n[Step 5] Analyzing Disease Candidates & Predefined Eligibility...")
    candidate_df, eligibility_records = analyze_disease_candidates(harmonized_dict, min_sample_threshold=500)
    proposed_set = select_proposed_disease_set(eligibility_records, target_k=7)
    
    dis_analysis_md = """# Disease Candidate Analysis & Eligibility Report

Evaluates disease candidates against strictly predefined clinical and experimental criteria:
1. **Sample Size:** $\\ge 500$ records across cohorts
2. **Reliable Label Mapping:** Unambiguous translation across SCP, SNOMED CT, and machine reports
3. **Cross-Dataset Representation:** Present in $\\ge 2$ cohorts (ideally all 3)
4. **ECG Observability:** Manifests with definitive physical electrodynamic hallmarks
5. **Evidence Verifiability:** Hallmarks are identifiable within anatomically defined lead subsets

*Zero model accuracy or downstream performance metrics were used to select diseases.*

---

## Cross-Dataset Candidate Disease Table

| Disease | PTB-XL Records (Pts) | Chapman Records (Pts) | MIMIC Records (Pts) | Total Records | Total Patients | Cross-Dataset | Eligible |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, r in candidate_df.iterrows():
        dis_analysis_md += f"| **{r['Disease']}** | {r['PTB-XL Records']} | {r['Chapman Records']} | {r['MIMIC Records']} | {r['Total Records']:,} | {r['Total Patients']:,} | {r['Cross-Dataset']} | **{r['Eligible']}** |\n"

    dis_analysis_md += """
---

## Detailed Eligibility & Inclusion/Exclusion Rationale

"""
    for rec in eligibility_records:
        dis_analysis_md += f"### {rec['disease']}\n"
        dis_analysis_md += f"- **Eligible:** `{'YES' if rec['is_eligible'] else 'NO'}`\n"
        dis_analysis_md += f"- **Total Prevalence:** {rec['total_records']:,} records across {rec['total_patients']:,} patients ({rec['datasets_present']}/3 datasets)\n"
        dis_analysis_md += f"- **Lead Hallmarks:** {rec['lead_hallmarks']}\n"
        dis_analysis_md += f"- **Decision Rationale:** {rec['rationale']}\n"
        dis_analysis_md += f"- **Criteria Checklist:** Sample Size: {'PASS' if rec['c1_sample_size'] else 'FAIL'} | Reliable Mapping: {'PASS' if rec['c2_reliable_mapping'] else 'FAIL'} | Cross-Dataset: {'PASS' if rec['c3_cross_dataset'] else 'FAIL'} | Observability: {'PASS' if rec['c4_ecg_observability'] else 'FAIL'} | Verifiability: {'PASS' if rec['c5_evidence_verifiability'] else 'FAIL'}\n\n"

    dis_analysis_md += """---

## Proposed Final Frozen 6–7 Disease Set

The proposed frozen set provides balanced representation across distinct anatomical and electrophysiological mechanisms (Rhythm, Conduction, Ischemia, Rate), with clear lead-specific grounding boundaries essential for lead-ablation experiments:

| # | Disease Name | Category | Primary Lead Hallmarks | Total Records | Cross-Dataset | Rationale for Freezing |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for idx, p in enumerate(proposed_set, 1):
        dis_analysis_md += f"| **{idx}** | **{p['disease']}** | Clinical Target | `{p['lead_hallmarks']}` | {p['total_records']:,} | {p['datasets_present']}/3 | {p['rationale']} |\n"

    dis_analysis_md += """
### Architectural Justification for Proposed Frozen Set
1. **Normal ECG (Negative Control)**: Baseline physiological control essential for specificity evaluation under lead degradation.
2. **Atrial Fibrillation (Supraventricular Arrhythmia)**: Hallmark irregular RR intervals and absent P waves observable globally across rhythm leads (Lead II, V1).
3. **Right Bundle Branch Block (Right Ventricular Conduction)**: Pathognomonic terminal conduction delay (rsR') strictly localized to anterior/septal leads (V1, V2) with slurred S wave in lateral leads (I, aVL, V5-V6). Tests localized evidence elimination when V1-V2 are masked.
4. **Left Bundle Branch Block (Left Ventricular Conduction)**: Broad monophasic R wave localized to lateral leads (I, aVL, V5-V6) and deep QS in V1-V2. Tests localized evidence elimination when lateral leads are masked.
5. **Sinus Bradycardia (Sinus Node Dysfunction - Slow Rate)**: Global rate disturbance (< 60 bpm) observable on any active lead; tests global reasoning stability under extreme lead reduction.
6. **Sinus Tachycardia (Sinus Node Dysfunction - Fast Rate)**: Global rate disturbance (> 100 bpm) observable on any active lead; complements Sinus Bradycardia across the physiological spectrum.
7. **1st Degree AV Block (Atrioventricular Conduction Delay)**: Fixed PR interval prolongation (> 200 ms) detectable in Lead II, aVF, and V1.
*(Alternative Ischemic Candidate: **Myocardial Infarction [Inferior / Anteroseptal]**, which provides regional ST-elevation and pathological Q-waves strictly confined to inferior [II, III, aVF] or septal [V1-V4] leads).*
"""
    Path("reports/disease_candidate_analysis.md").write_text(dis_analysis_md, encoding="utf-8")
    print("Written reports/disease_candidate_analysis.md")
    print("\n==================================================")
    print("PHASE A PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_pipeline()
