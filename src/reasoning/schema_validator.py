"""
Clinical Schema Validation Layer (Module 4, Section 4.1)
========================================================
Validates extracted structured observations against the rigid clinical taxonomy
and enforces strict lead attribution rules.

Validation Constraints:
    1. Key Admissibility: Only fields in F = {rhythm, rate, PR, QRS, ST} are admitted.
    2. Value Vocabulary: All categorical values must exist in the clinical taxonomy.
    3. Observability Invariant: Masked leads (m_l[lead] == 0) CANNOT appear in active
       lead attribution or positive ST elevation/depression finding sets.
    4. Electrophysiological Sanity: Conflicting physiological combinations (e.g.,
       AFib with normal PR interval) are flagged and corrected.
"""

from typing import Dict, Any, List, Tuple
from src.reasoning.clinical_taxonomy import (
    LEAD_NAMES,
    RHYTHM_CLASSES,
    RATE_CLASSES,
    PR_CLASSES,
    QRS_CLASSES,
    ST_STATES,
)


class SchemaValidationLayer:
    """
    Enforces rigid medical taxonomy validation and lead attribution guarantees
    on observations extracted from the decoupled perception layer.
    """

    def __init__(self):
        self.valid_fields = {"rhythm", "rate", "PR", "QRS", "ST", "lead_attribution"}

    def validate_single_observation(
        self,
        finding: Dict[str, Any],
        lead_avail_mask: List[int],
    ) -> Tuple[Dict[str, Any], bool, List[str]]:
        """
        Validates and sanitizes a single extracted structured finding dictionary.

        Returns:
            sanitized_finding: Cleaned dictionary meeting clinical taxonomy
            is_valid: Boolean indicating whether observation passed validation
            audit_log: List of validation notes or corrections applied
        """
        audit_log = []
        is_valid = True

        active_leads = set([LEAD_NAMES[i] for i, m in enumerate(lead_avail_mask) if m == 1])
        masked_leads = set([LEAD_NAMES[i] for i, m in enumerate(lead_avail_mask) if m == 0])

        # 1. Field Admissibility Check
        for key in finding.keys():
            if key not in self.valid_fields:
                audit_log.append(f"Rejected unadmitted key: '{key}'")
                is_valid = False

        # 2. Rhythm Validation
        rhythm_data = finding.get("rhythm", {})
        rhythm_val = rhythm_data.get("value")
        if rhythm_val not in RHYTHM_CLASSES:
            audit_log.append(f"Invalid rhythm vocabulary: '{rhythm_val}' -> Defaulted to 'Other / Arrhythmia'")
            rhythm_data["value"] = "Other / Arrhythmia"
            is_valid = False

        # 3. Rate Validation
        rate_data = finding.get("rate", {})
        rate_cat = rate_data.get("category")
        if rate_cat not in RATE_CLASSES:
            audit_log.append(f"Invalid rate vocabulary: '{rate_cat}' -> Defaulted to 'Normal Rate (60-100 bpm)'")
            rate_data["category"] = "Normal Rate (60-100 bpm)"
            is_valid = False

        # 4. PR Interval Validation
        pr_data = finding.get("PR", {})
        pr_cat = pr_data.get("category")
        if pr_cat not in PR_CLASSES:
            audit_log.append(f"Invalid PR vocabulary: '{pr_cat}' -> Defaulted to 'Absent / Indeterminate'")
            pr_data["category"] = "Absent / Indeterminate"
            is_valid = False

        # Electrophysiological Sanity: AFib cannot have measurable normal PR
        if rhythm_data.get("value") == "Atrial fibrillation" and pr_cat == "Normal (120-200 ms)":
            audit_log.append("Physiological conflict: AFib has absent P waves; PR corrected to 'Absent / Indeterminate'")
            pr_data["category"] = "Absent / Indeterminate"

        # 5. QRS Duration Validation
        qrs_data = finding.get("QRS", {})
        qrs_cat = qrs_data.get("category")
        if qrs_cat not in QRS_CLASSES:
            audit_log.append(f"Invalid QRS vocabulary: '{qrs_cat}' -> Defaulted to 'Normal (<120 ms)'")
            qrs_data["category"] = "Normal (<120 ms)"
            is_valid = False

        # 6. ST Segment & Observability Invariant Enforcement
        st_data = finding.get("ST", {})
        elev_leads = st_data.get("elevation_leads", [])
        dep_leads = st_data.get("depression_leads", [])

        # Check for leakage of masked leads
        sanitized_elev = []
        for l in elev_leads:
            if l in active_leads:
                sanitized_elev.append(l)
            else:
                audit_log.append(f"Observability Violation: ST elevation in masked lead '{l}' purged.")
                is_valid = False

        sanitized_dep = []
        for l in dep_leads:
            if l in active_leads:
                sanitized_dep.append(l)
            else:
                audit_log.append(f"Observability Violation: ST depression in masked lead '{l}' purged.")
                is_valid = False

        st_data["elevation_leads"] = sanitized_elev
        st_data["elevation_present"] = len(sanitized_elev) > 0
        st_data["depression_leads"] = sanitized_dep
        st_data["depression_present"] = len(sanitized_dep) > 0

        # Construct sanitized validated dictionary
        sanitized = {
            "rhythm": {
                "value": rhythm_data["value"],
                "derived_leads": [l for l in rhythm_data.get("derived_leads", []) if l in active_leads],
            },
            "rate": {
                "category": rate_data["category"],
                "derived_leads": [l for l in ["II", "V1"] if l in active_leads],
            },
            "PR": {
                "category": pr_data["category"],
                "derived_leads": [l for l in ["II", "V1"] if l in active_leads],
            },
            "QRS": {
                "category": qrs_data["category"],
                "derived_leads": [l for l in ["V1", "V6"] if l in active_leads],
            },
            "ST": st_data,
            "lead_attribution": {
                "active_leads": sorted(list(active_leads)),
                "masked_leads": sorted(list(masked_leads)),
                "observability_preserved": True,
            },
            "schema_version": "v1.0_strict_clinical_taxonomy",
        }

        return sanitized, is_valid, audit_log

    def validate_batch(
        self,
        findings: List[Dict[str, Any]],
        lead_avail_masks: List[List[int]],
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Validates batch of findings.
        Returns (sanitized_list, valid_count, modified_count).
        """
        sanitized_list = []
        valid_count = 0
        modified_count = 0

        for f, mask in zip(findings, lead_avail_masks):
            sanitized, is_val, log = self.validate_single_observation(f, mask)
            sanitized_list.append(sanitized)
            if is_val:
                valid_count += 1
            if len(log) > 0:
                modified_count += 1

        return sanitized_list, valid_count, modified_count
