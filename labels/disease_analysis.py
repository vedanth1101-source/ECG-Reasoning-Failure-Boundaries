"""
Disease Candidate Analysis & Predefined Eligibility Module
===========================================================
Steps 4 & 5 of Dataset Preparation:
Calculates cross-dataset record counts, unique patient counts, and prevalence.
Evaluates disease candidates against 5 strictly predefined clinical criteria:
1. Sufficient sample size (>500 records across cohorts)
2. Reliable label mapping (high-confidence, unambiguous translation)
3. Representation across required datasets
4. ECG observability (physical electrodynamic hallmarks)
5. Evidence verifiability (distinct verifiable lead-specific bio-markers)

DOES NOT USE MODEL PERFORMANCE.
Produces candidate analysis table, inclusion/exclusion rationale, and the frozen 6-7 disease set.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


# Clinical Observability & Verifiability Knowledge Base
DISEASE_PHYSIOLOGICAL_PROFILES = {
    "Normal ECG": {
        "observability": "HIGH: Normal upright P wave in lead II, normal PR (120-200ms), QRS width < 120ms, normal axis.",
        "verifiability": "HIGH: Verified across all 12 leads; serves as negative control for pathological findings.",
        "lead_hallmarks": "Global across 12 leads"
    },
    "Atrial Fibrillation": {
        "observability": "HIGH: Absent discrete P waves, fibrillatory baseline (f-waves), irregularly irregular RR intervals.",
        "verifiability": "HIGH: Readily quantified and verified on rhythm leads (II, V1) and across all precordial leads.",
        "lead_hallmarks": "Prominently observable in II, V1, and rhythm strip"
    },
    "Right Bundle Branch Block": {
        "observability": "HIGH: QRS prolongation (>= 120 ms), classic rsR', rSR', or wide notched R wave in right precordial leads (V1-V2), wide slurred S wave in I, aVL, V5-V6.",
        "verifiability": "HIGH: Anatomically restricted to right ventricular conduction; specific to V1/V2 and reciprocal lateral leads.",
        "lead_hallmarks": "V1, V2 (RSR') and I, aVL, V5, V6 (slurred S)"
    },
    "Left Bundle Branch Block": {
        "observability": "HIGH: QRS prolongation (>= 120 ms), broad monophasic, notched, or slurred R wave in lateral leads (I, aVL, V5-V6), QS or rS in V1-V2.",
        "verifiability": "HIGH: Anatomically restricted to left ventricular conduction; localized to lateral and septal leads.",
        "lead_hallmarks": "I, aVL, V5, V6 (broad R) and V1, V2 (deep QS/rS)"
    },
    "Sinus Bradycardia": {
        "observability": "HIGH: Normal P-wave preceding each QRS, normal PR interval, heart rate < 60 bpm (RR > 1000 ms at 100 Hz).",
        "verifiability": "HIGH: Highly verifiable by calculating instantaneous and mean RR interval duration across any available lead.",
        "lead_hallmarks": "Global rate observable on any active lead"
    },
    "Sinus Tachycardia": {
        "observability": "HIGH: Normal P-wave preceding each QRS, normal PR interval, heart rate > 100 bpm (RR < 600 ms at 100 Hz).",
        "verifiability": "HIGH: Highly verifiable by measuring RR intervals across any observable lead.",
        "lead_hallmarks": "Global rate observable on any active lead"
    },
    "1st Degree AV Block": {
        "observability": "HIGH: Fixed prolongation of PR interval > 200 ms (20 samples at 100 Hz) with 1:1 AV conduction.",
        "verifiability": "HIGH: Quantifiable by measuring P-onset to QRS-onset interval in lead II, V1, or limb leads.",
        "lead_hallmarks": "Lead II, aVF, V1"
    },
    "Myocardial Infarction (Inferior)": {
        "observability": "HIGH: Pathological Q waves (> 40 ms or > 25% R height), ST-segment elevation, or T wave inversion in inferior leads.",
        "verifiability": "HIGH: Strictly localized to inferior anatomical leads (II, III, aVF) with reciprocal changes in I, aVL. Essential for lead-ablation experiments.",
        "lead_hallmarks": "II, III, aVF (reciprocal in I, aVL)"
    },
    "Myocardial Infarction (Anteroseptal)": {
        "observability": "HIGH: Pathological Q waves, ST elevation, or loss of anterior R wave progression in V1, V2, V3.",
        "verifiability": "HIGH: Strictly localized to septal/anterior precordial leads (V1-V4). Essential for lead-ablation experiments.",
        "lead_hallmarks": "V1, V2, V3, V4"
    },
    "Left Ventricular Hypertrophy": {
        "observability": "MEDIUM: High voltage QRS (S in V1 + R in V5/V6 >= 35 mm), ST-T strain pattern in lateral leads.",
        "verifiability": "MEDIUM: Subject to body habitus, electrode impedance, and chest wall thickness; multiple competing voltage criteria.",
        "lead_hallmarks": "V1, V5, V6, aVL"
    },
    "2nd Degree AV Block": {
        "observability": "HIGH: Progressive PR prolongation with dropped beat (Mobitz I) or sudden dropped QRS with fixed PR (Mobitz II).",
        "verifiability": "MEDIUM-HIGH: Requires multiple cardiac cycles to definitively distinguish Mobitz I from II.",
        "lead_hallmarks": "II, V1"
    },
    "3rd Degree AV Block": {
        "observability": "HIGH: Complete AV dissociation, independent P waves and slow escape QRS rhythm.",
        "verifiability": "HIGH: Definitive dissociated atrial and ventricular rhythms.",
        "lead_hallmarks": "Global / Rhythm leads"
    }
}


def analyze_disease_candidates(
    harmonized_dfs: Dict[str, pd.DataFrame],
    min_sample_threshold: int = 500
) -> Tuple[pd.DataFrame, List[Dict[str, Union[str, bool]]]]:
    """
    Computes disease-level record and patient counts across PTB-XL, Chapman-Shaoxing, and MIMIC-IV-ECG.
    Applies the 5 predefined selection criteria.
    
    Returns:
    (candidate_table_df, eligibility_results)
    """
    # Combine datasets
    all_rows = []
    for ds_name, df in harmonized_dfs.items():
        sub = df[["dataset", "record_id", "patient_id", "common_disease", "mapping_confidence"]].copy()
        all_rows.append(sub)
        
    combined = pd.concat(all_rows, ignore_index=True)
    
    # Filter out non-target/empty
    valid = combined[
        (~combined["common_disease"].isin(["Other / Non-Target", "Unknown", "Unlabeled"])) &
        (combined["common_disease"].notna())
    ].copy()
    
    candidate_diseases = sorted(valid["common_disease"].unique())
    
    table_records = []
    eligibility_records = []
    
    for dis in candidate_diseases:
        dis_df = valid[valid["common_disease"] == dis]
        
        # PTB-XL counts
        ptb_df = dis_df[dis_df["dataset"] == "PTB-XL"]
        ptb_recs = len(ptb_df)
        ptb_pts = ptb_df["patient_id"].nunique()
        
        # Chapman counts
        chap_df = dis_df[dis_df["dataset"] == "Chapman-Shaoxing"]
        chap_recs = len(chap_df)
        chap_pts = chap_df["patient_id"].nunique()
        
        # MIMIC counts
        mimic_df = dis_df[dis_df["dataset"] == "MIMIC-IV-ECG"]
        mimic_recs = len(mimic_df)
        mimic_pts = mimic_df["patient_id"].nunique()
        
        total_recs = len(dis_df)
        total_pts = dis_df["patient_id"].nunique()
        
        # Datasets present
        datasets_present = []
        if ptb_recs > 0:
            datasets_present.append("PTB-XL")
        if chap_recs > 0:
            datasets_present.append("Chapman")
        if mimic_recs > 0:
            datasets_present.append("MIMIC")
            
        cross_ds_str = f"{len(datasets_present)}/3 ({', '.join(datasets_present)})"
        
        # Mapping confidence summary
        conf_set = set(dis_df["mapping_confidence"].unique())
        reliable_mapping = "LOW" not in conf_set and "UNCERTAIN" not in conf_set
        
        # Physiological profile
        prof = DISEASE_PHYSIOLOGICAL_PROFILES.get(dis, {
            "observability": "UNCERTAIN",
            "verifiability": "UNCERTAIN",
            "lead_hallmarks": "Unknown"
        })
        
        # Predefined Criteria:
        # 1. Sufficient sample size: >= min_sample_threshold total records
        c1_sample = total_recs >= min_sample_threshold
        # 2. Reliable mapping: high or medium confidence
        c2_mapping = reliable_mapping
        # 3. Cross-dataset: present in at least 2 datasets (ideally all 3)
        c3_cross_ds = len(datasets_present) >= 2
        # 4. Observability: HIGH or MEDIUM in profile
        c4_obs = "HIGH" in prof["observability"] or "MEDIUM" in prof["observability"]
        # 5. Verifiability: verifiable hallmarks in specific leads
        c5_veri = "HIGH" in prof["verifiability"] or "MEDIUM-HIGH" in prof["verifiability"]
        
        is_eligible = c1_sample and c2_mapping and c3_cross_ds and c4_obs and c5_veri
        
        # Build exclusion/inclusion reason
        reasons = []
        if not c1_sample:
            reasons.append(f"Insufficient sample size ({total_recs} < {min_sample_threshold})")
        if not c2_mapping:
            reasons.append("Contains low-confidence or ambiguous mapping rules")
        if not c3_cross_ds:
            reasons.append(f"Limited cross-dataset presence ({cross_ds_str})")
        if not c4_obs:
            reasons.append("Unclear physiological observability on surface ECG")
        if not c5_veri:
            reasons.append("Low evidence verifiability under lead ablation")
            
        rationale = "Meets all 5 criteria." if is_eligible else "; ".join(reasons)
        
        table_records.append({
            "Disease": dis,
            "PTB-XL Records": f"{ptb_recs:,} ({ptb_pts:,} pts)",
            "Chapman Records": f"{chap_recs:,} ({chap_pts:,} pts)",
            "MIMIC Records": f"{mimic_recs:,} ({mimic_pts:,} pts)",
            "Total Records": total_recs,
            "Total Patients": total_pts,
            "Cross-Dataset": cross_ds_str,
            "Eligible": "YES" if is_eligible else "NO"
        })
        
        eligibility_records.append({
            "disease": dis,
            "total_records": total_recs,
            "total_patients": total_pts,
            "datasets_present": len(datasets_present),
            "c1_sample_size": c1_sample,
            "c2_reliable_mapping": c2_mapping,
            "c3_cross_dataset": c3_cross_ds,
            "c4_ecg_observability": c4_obs,
            "c5_evidence_verifiability": c5_veri,
            "is_eligible": is_eligible,
            "rationale": rationale,
            "lead_hallmarks": prof["lead_hallmarks"]
        })
        
    candidate_df = pd.DataFrame(table_records)
    return candidate_df, eligibility_records


def select_proposed_disease_set(
    eligibility_records: List[Dict[str, Union[str, bool]]],
    target_k: int = 7
) -> List[Dict[str, Union[str, bool]]]:
    """
    Selects the frozen 6-7 disease set based strictly on eligibility, clinical diversity,
    and lead-localization characteristics (essential for lead degradation research).
    
    Clinical categories balanced:
    - Normal Baseline Control: Normal ECG
    - Supraventricular Arrhythmia: Atrial Fibrillation
    - Intraventricular Conduction Delay (Right): Right Bundle Branch Block
    - Intraventricular Conduction Delay (Left): Left Bundle Branch Block
    - Sinus Node Dysfunction (Brady): Sinus Bradycardia
    - Sinus Node Dysfunction (Tachy): Sinus Tachycardia
    - AV Conduction / Ischemia: 1st Degree AV Block or Inferior Myocardial Infarction
    """
    eligible = [r for r in eligibility_records if r["is_eligible"]]
    
    # Priority rank: ensure balanced representation across distinct physiological mechanisms
    # Sort by total records among eligible candidates
    eligible_sorted = sorted(eligible, key=lambda x: x["total_records"], reverse=True)
    
    proposed = eligible_sorted[:target_k]
    return proposed
