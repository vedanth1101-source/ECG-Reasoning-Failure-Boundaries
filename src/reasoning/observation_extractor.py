"""
Linear Observation Extraction Heads (Module 4, Equations 4.6 & 4.7)
===================================================================
Projects the continuous latent representation Z_l in R^{768} through linear
projection heads into observation logits:
    z^(f)_l = W_f Z_l + b_f            (Equation 4.6)
    y^(f)_l = argmax softmax(z^(f)_l)  (Equation 4.7)
for each field f in the fixed clinical field set:
    F = {rhythm, rate, PR, QRS, ST}

This decoupled perception layer allows failures of perception to be distinguished
from failures of inference, providing intermediate inspectable claims.
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.reasoning.clinical_taxonomy import (
    LEAD_NAMES,
    LEAD_TO_IDX,
    RHYTHM_CLASSES,
    RATE_CLASSES,
    PR_CLASSES,
    QRS_CLASSES,
    ST_STATES,
)


class LinearObservationExtractor(nn.Module):
    """
    Decoupled perception layer: Multi-head linear projection mapping
    latent vector Z_l in R^{768} to discrete structured observations O_l.
    """

    def __init__(self, d_in: int = 768):
        super().__init__()
        self.d_in = d_in

        # Equation 4.6: Dedicated linear heads W_f Z_l + b_f
        # 1. Rhythm Head: 5 classes (Sinus, AFib, Sinus Brady, Sinus Tachy, Other)
        self.head_rhythm = nn.Linear(d_in, len(RHYTHM_CLASSES))

        # 2. Rate Head: 3 classes (Bradycardia <60, Normal 60-100, Tachycardia >100)
        self.head_rate = nn.Linear(d_in, len(RATE_CLASSES))

        # 3. PR Interval Head: 3 classes (Normal 120-200ms, Prolonged >200ms, Absent/Indeterminate)
        self.head_pr = nn.Linear(d_in, len(PR_CLASSES))

        # 4. QRS Duration Head: 2 classes (Normal <120ms, Prolonged >=120ms)
        self.head_qrs = nn.Linear(d_in, len(QRS_CLASSES))

        # 5. ST Segment Head: 12 leads x 3 states (Isoelectric, Elevation, Depression)
        self.head_st = nn.Linear(d_in, len(LEAD_NAMES) * len(ST_STATES))

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes observation logits z^(f)_l across all 5 fields.

        Args:
            z: Continuous latent representation (B, 768)

        Returns:
            Dictionary of raw logits per field:
                - 'rhythm': (B, 5)
                - 'rate': (B, 3)
                - 'PR': (B, 3)
                - 'QRS': (B, 2)
                - 'ST': (B, 12, 3)
        """
        B = z.size(0)
        z_rhythm = self.head_rhythm(z)
        z_rate = self.head_rate(z)
        z_pr = self.head_pr(z)
        z_qrs = self.head_qrs(z)
        z_st = self.head_st(z).view(B, len(LEAD_NAMES), len(ST_STATES))

        return {
            "rhythm": z_rhythm,
            "rate": z_rate,
            "PR": z_pr,
            "QRS": z_qrs,
            "ST": z_st,
        }

    @staticmethod
    def to_context_builder_format(finding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a clinical taxonomy observation dictionary into the standardized format
        directly consumable by ControlContextBuilder.
        """
        # 1. Rhythm
        rhythm_data = finding.get("rhythm", {})
        if isinstance(rhythm_data, dict):
            rhythm_val = rhythm_data.get("value", "Unknown / Undetermined")
        else:
            rhythm_val = str(rhythm_data)

        # 2. Heart rate
        rate_data = finding.get("rate", {})
        if "heart_rate" in finding and finding["heart_rate"] is not None:
            hr_val = float(finding["heart_rate"])
        elif isinstance(rate_data, dict):
            cat = rate_data.get("category", "")
            if "Bradycardia" in cat:
                hr_val = 50.0
            elif "Tachycardia" in cat:
                hr_val = 115.0
            else:
                hr_val = 75.0
        else:
            hr_val = 75.0

        # 3. PR interval
        pr_data = finding.get("PR", {})
        if "pr_interval_ms" in finding:
            pr_val = finding["pr_interval_ms"]
        elif isinstance(pr_data, dict):
            cat = pr_data.get("category", "")
            if "Prolonged" in cat:
                pr_val = 240.0
            elif "Normal" in cat:
                pr_val = 160.0
            else:
                pr_val = None
        else:
            pr_val = 160.0

        # 4. QRS duration
        qrs_data = finding.get("QRS", {})
        if "qrs_duration_ms" in finding and finding["qrs_duration_ms"] is not None:
            qrs_val = float(finding["qrs_duration_ms"])
        elif isinstance(qrs_data, dict):
            cat = qrs_data.get("category", "")
            if "Prolonged" in cat:
                qrs_val = 140.0
            else:
                qrs_val = 90.0
        else:
            qrs_val = 90.0

        # 5. ST findings
        st_data = finding.get("ST", {})
        if "st_elevation" in finding and isinstance(finding["st_elevation"], dict):
            st_elevation = finding["st_elevation"]
        elif isinstance(st_data, dict):
            elev_leads = st_data.get("elevation_leads", [])
            st_elevation = {
                "present": len(elev_leads) > 0,
                "leads": elev_leads,
                "amplitude_mm": {l: 1.5 for l in elev_leads},
            }
        else:
            st_elevation = {"present": False, "leads": [], "amplitude_mm": {}}

        if "st_depression" in finding and isinstance(finding["st_depression"], dict):
            st_depression = finding["st_depression"]
        elif isinstance(st_data, dict):
            dep_leads = st_data.get("depression_leads", [])
            st_depression = {
                "present": len(dep_leads) > 0,
                "leads": dep_leads,
                "amplitude_mm": {l: -1.0 for l in dep_leads},
            }
        else:
            st_depression = {"present": False, "leads": [], "amplitude_mm": {}}

        # 6. T wave inversion
        if "t_wave_inversion" in finding and isinstance(finding["t_wave_inversion"], dict):
            t_inv = finding["t_wave_inversion"]
        elif "T" in finding and isinstance(finding["T"], dict):
            t_leads = finding["T"].get("inversion_leads", [])
            t_inv = {"present": len(t_leads) > 0, "leads": t_leads}
        else:
            t_inv = {"present": False, "leads": []}

        return {
            "rhythm": rhythm_val,
            "heart_rate": hr_val,
            "pr_interval_ms": pr_val,
            "qrs_duration_ms": qrs_val,
            "st_elevation": st_elevation,
            "st_depression": st_depression,
            "t_wave_inversion": t_inv,
            "rate": rate_data if isinstance(rate_data, dict) else {"category": "Normal Rate (60-100 bpm)"},
            "PR": pr_data if isinstance(pr_data, dict) else {"category": "Normal (120-200 ms)"},
            "QRS": qrs_data if isinstance(qrs_data, dict) else {"category": "Normal (<120 ms)"},
            "ST": st_data if isinstance(st_data, dict) else {},
            "lead_attribution": finding.get("lead_attribution", {}),
        }

    def predict_discretized(
        self,
        z: torch.Tensor,
        lead_avail_mask: Optional[torch.Tensor] = None,
        compatible_format: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Computes Equation 4.7: y^(f)_l = argmax softmax(z^(f)_l)
        and converts predictions into structured findings O_l.

        Args:
            z: Continuous latent representation (B, 768)
            lead_avail_mask: Binary availability mask (B, 12)
            compatible_format: If True, returns findings in flat format compatible
                               with ControlContextBuilder. If False, returns rigid
                               taxonomy dictionary format.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(z)

            # Argmax discretization
            pred_rhythm_idx = torch.argmax(F.softmax(logits["rhythm"], dim=-1), dim=-1).cpu().numpy()
            pred_rate_idx = torch.argmax(F.softmax(logits["rate"], dim=-1), dim=-1).cpu().numpy()
            pred_pr_idx = torch.argmax(F.softmax(logits["PR"], dim=-1), dim=-1).cpu().numpy()
            pred_qrs_idx = torch.argmax(F.softmax(logits["QRS"], dim=-1), dim=-1).cpu().numpy()
            pred_st_idx = torch.argmax(F.softmax(logits["ST"], dim=-1), dim=-1).cpu().numpy()  # (B, 12)

        B = z.size(0)
        masks_np = (
            lead_avail_mask.cpu().numpy()
            if lead_avail_mask is not None
            else torch.ones(B, 12).numpy()
        )

        extracted_findings = []
        for b in range(B):
            mask_b = masks_np[b]
            active_lead_names = [LEAD_NAMES[i] for i, m in enumerate(mask_b) if m == 1]

            # Rhythm finding
            rhythm_val = RHYTHM_CLASSES[pred_rhythm_idx[b]]

            # Rate finding
            rate_val = RATE_CLASSES[pred_rate_idx[b]]

            # PR interval finding
            pr_val = PR_CLASSES[pred_pr_idx[b]]

            # QRS duration finding
            qrs_val = QRS_CLASSES[pred_qrs_idx[b]]

            # ST Segment findings with explicit lead attribution
            st_elevation_leads = []
            st_depression_leads = []
            st_by_lead = {}

            for l_idx, l_name in enumerate(LEAD_NAMES):
                # Observability constraint: masked leads are strictly isoelectric / unavailable
                if mask_b[l_idx] == 0:
                    st_by_lead[l_name] = "UNAVAILABLE_MASKED"
                    continue

                st_state_idx = pred_st_idx[b, l_idx]
                st_state_str = ST_STATES[st_state_idx]
                st_by_lead[l_name] = st_state_str

                if st_state_str == "ST_Elevation":
                    st_elevation_leads.append(l_name)
                elif st_state_str == "ST_Depression":
                    st_depression_leads.append(l_name)

            sample_finding = {
                "rhythm": {
                    "value": rhythm_val,
                    "derived_leads": active_lead_names if len(active_lead_names) > 0 else [],
                },
                "rate": {
                    "category": rate_val,
                    "derived_leads": [l for l in ["II", "V1"] if l in active_lead_names],
                },
                "PR": {
                    "category": pr_val,
                    "derived_leads": [l for l in ["II", "V1"] if l in active_lead_names],
                },
                "QRS": {
                    "category": qrs_val,
                    "derived_leads": [l for l in ["V1", "V6"] if l in active_lead_names],
                },
                "ST": {
                    "elevation_present": len(st_elevation_leads) > 0,
                    "elevation_leads": st_elevation_leads,
                    "depression_present": len(st_depression_leads) > 0,
                    "depression_leads": st_depression_leads,
                    "lead_states": st_by_lead,
                },
                "lead_attribution": {
                    "active_leads": active_lead_names,
                    "masked_leads": [LEAD_NAMES[i] for i, m in enumerate(mask_b) if m == 0],
                },
            }

            if compatible_format:
                extracted_findings.append(self.to_context_builder_format(sample_finding))
            else:
                extracted_findings.append(sample_finding)

        return extracted_findings
