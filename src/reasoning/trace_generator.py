"""
Structured Reasoning Trace Generator for Control Path
======================================================
Transforms raw outputs of ControlReasoningTransformer (posteriors and cross-attention
matrix) into structured clinical reasoning traces adhering to Phase 4 specifications.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch

from src.reasoning.context_builder import CANDIDATE_ACRONYMS, CANDIDATE_NAMES, LEAD_NAMES


class StructuredTraceGenerator:
    """
    Emits formal structured reasoning traces for each evaluated ECG recording.
    """

    def __init__(
        self,
        evidence_weight_threshold: float = 0.05,
        top_evidence_k: int = 5,
    ):
        self.evidence_weight_threshold = evidence_weight_threshold
        self.top_evidence_k = top_evidence_k
        self.acronym_to_name = dict(zip(CANDIDATE_ACRONYMS, CANDIDATE_NAMES))

    def _extract_leads_from_token_name(self, token_name: str) -> List[str]:
        """Extracts associated lead names from a token label if present."""
        leads = []
        for l in LEAD_NAMES:
            if token_name.endswith(f"_{l}"):
                leads.append(l)
        return leads

    def generate_single_trace(
        self,
        probabilities: np.ndarray,              # (8,)
        attention_matrix: np.ndarray,           # (8, N_evidence)
        evidence_names: List[str],              # (N_evidence,)
        lead_avail_mask: np.ndarray,            # (12,)
        observation: Dict[str, Any],
        record_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates a structured reasoning trace for a single ECG recording.
        """
        active_leads = [LEAD_NAMES[i] for i, m in enumerate(lead_avail_mask) if m == 1]
        masked_leads = [LEAD_NAMES[i] for i, m in enumerate(lead_avail_mask) if m == 0]

        # Determine top hypothesis
        top_idx = int(np.argmax(probabilities))
        top_acronym = CANDIDATE_ACRONYMS[top_idx]
        top_disease_name = self.acronym_to_name[top_acronym]
        top_confidence = float(probabilities[top_idx])
        uncertainty = float(1.0 - top_confidence)

        # Build posterior dictionary across all 8 candidates
        posterior_dict = {
            acronym: round(float(probabilities[i]), 4)
            for i, acronym in enumerate(CANDIDATE_ACRONYMS)
        }

        # Extract attention weights for the top hypothesis
        top_hypo_attn = attention_matrix[top_idx]  # (N_evidence,)

        # Rank evidence tokens by attention weight
        sorted_indices = np.argsort(top_hypo_attn)[::-1]

        supporting_evidence = []
        contradicting_evidence = []

        for idx in sorted_indices:
            weight = float(top_hypo_attn[idx])
            if weight < self.evidence_weight_threshold:
                continue

            token_name = evidence_names[idx]
            assoc_leads = self._extract_leads_from_token_name(token_name)

            # Check lead observability
            is_observable = True
            if assoc_leads:
                is_observable = all(l in active_leads for l in assoc_leads)

            evidence_item = {
                "token_id": token_name,
                "attention_weight": round(weight, 4),
                "associated_leads": assoc_leads,
                "is_observable_in_mask": is_observable,
            }

            supporting_evidence.append(evidence_item)
            if len(supporting_evidence) >= self.top_evidence_k:
                break

        # Check for physiological contradictions if observable
        hr = observation.get("heart_rate")
        if hr is not None and "II" in active_leads:
            if top_acronym == "SBRAD" and hr >= 60:
                contradicting_evidence.append({
                    "finding": f"Heart rate is {hr} bpm (>= 60 bpm contradicts Sinus Bradycardia)",
                    "associated_leads": ["II"],
                })
            elif top_acronym == "STACH" and hr <= 100:
                contradicting_evidence.append({
                    "finding": f"Heart rate is {hr} bpm (<= 100 bpm contradicts Sinus Tachycardia)",
                    "associated_leads": ["II"],
                })

        return {
            "record_id": record_id,
            "path_type": "CONTROL_ECG_ONLY",
            "top_hypothesis": {
                "acronym": top_acronym,
                "disease_name": top_disease_name,
                "confidence": round(top_confidence, 4),
            },
            "posterior_probabilities": posterior_dict,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "uncertainty": round(uncertainty, 4),
            "lead_observability": {
                "active_lead_count": len(active_leads),
                "active_leads": active_leads,
                "masked_leads": masked_leads,
            },
        }

    def generate_batch_traces(
        self,
        probabilities: torch.Tensor,            # (B, 8)
        attention_matrix: torch.Tensor,         # (B, 8, N_evidence)
        evidence_names: List[str],
        lead_avail_masks: torch.Tensor,         # (B, 12)
        observations: List[Dict[str, Any]],
        record_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Batch generation of structured reasoning traces."""
        probs_np = probabilities.detach().cpu().numpy() if isinstance(probabilities, torch.Tensor) else np.asarray(probabilities)
        attn_np = attention_matrix.detach().cpu().numpy() if isinstance(attention_matrix, torch.Tensor) else np.asarray(attention_matrix)
        masks_np = lead_avail_masks.detach().cpu().numpy() if isinstance(lead_avail_masks, torch.Tensor) else np.asarray(lead_avail_masks)

        traces = []
        B = probs_np.shape[0]
        for b in range(B):
            rec_id = record_ids[b] if record_ids is not None else f"rec_{b:05d}"
            trace = self.generate_single_trace(
                probabilities=probs_np[b],
                attention_matrix=attn_np[b],
                evidence_names=evidence_names,
                lead_avail_mask=masks_np[b],
                observation=observations[b],
                record_id=rec_id,
            )
            traces.append(trace)
        return traces
