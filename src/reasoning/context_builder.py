"""
Context Construction for Control Path (ECG-Only)
=================================================
Builds multimodal evidence and candidate query tokens strictly from:
    1. Continuous latent embedding Z_ECG (in R^{768})
    2. Discrete structured clinical observations O_l (rhythm, HR, PR, QRS, ST, T)
    3. Lead availability mask m_l (in {0, 1}^12)
    4. Eight frozen candidate disease queries H_i

STRICT CONTROL PATH INVARIANT:
Zero external Clinical Knowledge Prior (CKP) tokens are accepted or constructed.
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_TO_IDX = {name: idx for idx, name in enumerate(LEAD_NAMES)}

CANDIDATE_ACRONYMS = ["NORM", "AFIB", "RBBB", "LBBB", "SBRAD", "STACH", "1AVB", "MI"]
CANDIDATE_NAMES = [
    "Normal ECG",
    "Atrial Fibrillation",
    "Right Bundle Branch Block",
    "Left Bundle Branch Block",
    "Sinus Bradycardia",
    "Sinus Tachycardia",
    "First Degree AV Block",
    "Myocardial Infarction",
]

KNOWN_RHYTHMS = [
    "Sinus rhythm",
    "Atrial fibrillation",
    "Sinus tachycardia",
    "Sinus bradycardia",
    "Supraventricular tachycardia",
    "Unknown / Undetermined",
]
RHYTHM_TO_IDX = {r: i for i, r in enumerate(KNOWN_RHYTHMS)}


class ControlContextBuilder(nn.Module):
    """
    Constructs token sequences for the Control Path (ECG-Only).
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ecg: int = 768,
        num_candidates: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ecg = d_ecg
        self.num_candidates = num_candidates

        # 1. Continuous ECG embedding projection
        self.ecg_proj = nn.Sequential(
            nn.Linear(d_ecg, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Lead positional and availability status embeddings
        self.lead_pos_embed = nn.Embedding(len(LEAD_NAMES), d_model)
        self.lead_status_embed = nn.Embedding(2, d_model)  # 0: masked, 1: active

        # 3. Observation feature encoders
        self.rhythm_embed = nn.Embedding(len(KNOWN_RHYTHMS) + 1, d_model)
        # Numerical intervals: [HR, PR, QRS] normalized
        self.interval_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Morphological wave abnormality embeddings
        self.st_elevation_embed = nn.Embedding(len(LEAD_NAMES), d_model)
        self.st_depression_embed = nn.Embedding(len(LEAD_NAMES), d_model)
        self.t_inversion_embed = nn.Embedding(len(LEAD_NAMES), d_model)

        # 4. Eight frozen candidate disease query tokens (learned queries)
        self.candidate_queries = nn.Parameter(
            torch.randn(num_candidates, d_model) / (d_model ** 0.5)
        )

        # Layer normalization for token stabilization
        self.norm = nn.LayerNorm(d_model)

    def _normalize_intervals(
        self,
        hr: Optional[float],
        pr: Optional[float],
        qrs: Optional[float],
    ) -> torch.Tensor:
        """Standardizes interval metrics: HR in [30, 200], PR in [80, 400], QRS in [50, 250]."""
        hr_val = (float(hr) - 75.0) / 30.0 if hr is not None else 0.0
        pr_val = (float(pr) - 160.0) / 40.0 if pr is not None else 0.0
        qrs_val = (float(qrs) - 100.0) / 25.0 if qrs is not None else 0.0
        return torch.tensor([hr_val, pr_val, qrs_val], dtype=torch.float32)

    def forward(
        self,
        z_ecg: torch.Tensor,
        observations: List[Dict[str, Any]],
        lead_avail_mask: torch.Tensor,
        ckp_tokens: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
        """
        Builds the Control Path token sequence.

        Args:
            z_ecg: Continuous latent representation (B, 768)
            observations: List of B structured observation dicts
            lead_avail_mask: Binary availability mask (B, 12)
            ckp_tokens: MUST BE NONE. Enforces Control Path isolation.

        Returns:
            evidence_tokens: (B, N_evidence, d_model)
            evidence_mask: (B, N_evidence) True for valid tokens, False for padding
            evidence_token_names: List of N_evidence token identifier labels
            query_tokens: (B, num_candidates, d_model)
        """
        # Strict Control Path invariant check
        if ckp_tokens is not None:
            raise ValueError(
                "[CONTROL PATH VIOLATION] CKP tokens detected in Control Path. "
                "Control Path must operate strictly on ECG signals and observations without CKP."
            )

        B = z_ecg.size(0)
        device = z_ecg.device

        # Token 0: Continuous ECG embedding
        token_ecg = self.ecg_proj(z_ecg).unsqueeze(1)  # (B, 1, d_model)

        # Tokens 1-12: Lead status tokens (representing active/masked sensor geometry)
        lead_indices = torch.arange(len(LEAD_NAMES), device=device)
        lead_pos = self.lead_pos_embed(lead_indices).unsqueeze(0).expand(B, -1, -1)  # (B, 12, d)
        mask_binary = lead_avail_mask.long().to(device)  # (B, 12)
        lead_status = self.lead_status_embed(mask_binary)  # (B, 12, d)
        tokens_leads = lead_pos + lead_status  # (B, 12, d)

        # Batch encode structured observations (supports both flat and taxonomy schemas)
        rhythm_ids = []
        interval_feats = []

        for obs in observations:
            # 1. Rhythm: string or {"value": ...}
            rhythm_entry = obs.get("rhythm", "Unknown / Undetermined")
            if isinstance(rhythm_entry, dict):
                rhythm_str = rhythm_entry.get("value", "Unknown / Undetermined")
            else:
                rhythm_str = str(rhythm_entry)
            r_idx = RHYTHM_TO_IDX.get(rhythm_str, len(KNOWN_RHYTHMS))
            rhythm_ids.append(r_idx)

            # 2. Heart rate: numerical or category fallback
            hr = obs.get("heart_rate")
            if hr is None and "rate" in obs and isinstance(obs["rate"], dict):
                cat = obs["rate"].get("category", "")
                hr = 50.0 if "Bradycardia" in cat else (115.0 if "Tachycardia" in cat else 75.0)

            # 3. PR interval: numerical or category fallback
            pr = obs.get("pr_interval_ms")
            if pr is None and "PR" in obs and isinstance(obs["PR"], dict):
                cat = obs["PR"].get("category", "")
                pr = 240.0 if "Prolonged" in cat else (160.0 if "Normal" in cat else None)

            # 4. QRS duration: numerical or category fallback
            qrs = obs.get("qrs_duration_ms")
            if qrs is None and "QRS" in obs and isinstance(obs["QRS"], dict):
                cat = obs["QRS"].get("category", "")
                qrs = 140.0 if "Prolonged" in cat else 90.0

            interval_feats.append(self._normalize_intervals(hr, pr, qrs))

        rhythm_tensor = torch.tensor(rhythm_ids, device=device)  # (B,)
        tokens_rhythm = self.rhythm_embed(rhythm_tensor).unsqueeze(1)  # (B, 1, d)

        interval_tensor = torch.stack(interval_feats).to(device)  # (B, 3)
        tokens_intervals = self.interval_proj(interval_tensor).unsqueeze(1)  # (B, 1, d)

        # Morphological tokens (ST elevation, ST depression, T wave inversion)
        # Fixed slots: 12 leads for ST elevation, 12 for ST depression, 12 for T inversion
        st_elev_tokens_list = []
        st_dep_tokens_list = []
        t_inv_tokens_list = []

        for b_idx, obs in enumerate(observations):
            # ST Elevation (supports both "st_elevation" and "ST" format)
            if "st_elevation" in obs and isinstance(obs["st_elevation"], dict):
                st_elev = obs["st_elevation"]
                elev_leads = set(st_elev.get("leads", [])) if st_elev.get("present") else set()
            elif "ST" in obs and isinstance(obs["ST"], dict):
                st_elev = obs["ST"]
                elev_leads = set(st_elev.get("elevation_leads", [])) if st_elev.get("elevation_present") else set()
            else:
                elev_leads = set()

            b_elev = []
            for l_idx, l_name in enumerate(LEAD_NAMES):
                # Only active if lead is both present in finding AND physically observable
                is_active = (l_name in elev_leads) and (lead_avail_mask[b_idx, l_idx].item() == 1)
                emb = self.st_elevation_embed(torch.tensor(l_idx, device=device))
                b_elev.append(emb if is_active else torch.zeros_like(emb))
            st_elev_tokens_list.append(torch.stack(b_elev))

            # ST Depression (supports both "st_depression" and "ST" format)
            if "st_depression" in obs and isinstance(obs["st_depression"], dict):
                st_dep = obs["st_depression"]
                dep_leads = set(st_dep.get("leads", [])) if st_dep.get("present") else set()
            elif "ST" in obs and isinstance(obs["ST"], dict):
                st_dep = obs["ST"]
                dep_leads = set(st_dep.get("depression_leads", [])) if st_dep.get("depression_present") else set()
            else:
                dep_leads = set()

            b_dep = []
            for l_idx, l_name in enumerate(LEAD_NAMES):
                is_active = (l_name in dep_leads) and (lead_avail_mask[b_idx, l_idx].item() == 1)
                emb = self.st_depression_embed(torch.tensor(l_idx, device=device))
                b_dep.append(emb if is_active else torch.zeros_like(emb))
            st_dep_tokens_list.append(torch.stack(b_dep))

            # T Wave Inversion (supports both "t_wave_inversion" and "T" format)
            if "t_wave_inversion" in obs and isinstance(obs["t_wave_inversion"], dict):
                t_inv = obs["t_wave_inversion"]
                t_leads = set(t_inv.get("leads", [])) if t_inv.get("present") else set()
            elif "T" in obs and isinstance(obs["T"], dict):
                t_inv = obs["T"]
                t_leads = set(t_inv.get("inversion_leads", [])) if t_inv.get("inversion_present") else set()
            else:
                t_leads = set()

            b_t = []
            for l_idx, l_name in enumerate(LEAD_NAMES):
                is_active = (l_name in t_leads) and (lead_avail_mask[b_idx, l_idx].item() == 1)
                emb = self.t_inversion_embed(torch.tensor(l_idx, device=device))
                b_t.append(emb if is_active else torch.zeros_like(emb))
            t_inv_tokens_list.append(torch.stack(b_t))

        tokens_st_elev = torch.stack(st_elev_tokens_list)  # (B, 12, d)
        tokens_st_dep = torch.stack(st_dep_tokens_list)    # (B, 12, d)
        tokens_t_inv = torch.stack(t_inv_tokens_list)      # (B, 12, d)

        # Concatenate all evidence tokens:
        # 1 (ECG) + 12 (Leads) + 1 (Rhythm) + 1 (Intervals) + 12 (STE) + 12 (STD) + 12 (TWI) = 51 tokens
        evidence_tokens = torch.cat(
            [
                token_ecg,
                tokens_leads,
                tokens_rhythm,
                tokens_intervals,
                tokens_st_elev,
                tokens_st_dep,
                tokens_t_inv,
            ],
            dim=1,
        )  # (B, 51, d_model)
        evidence_tokens = self.norm(evidence_tokens)

        # Generate canonical token identifier labels
        evidence_token_names = (
            ["ECG_CONTINUOUS_FM"]
            + [f"LEAD_OBS_{l}" for l in LEAD_NAMES]
            + ["RHYTHM_PATTERN"]
            + ["INTERVAL_METRICS"]
            + [f"ST_ELEV_{l}" for l in LEAD_NAMES]
            + [f"ST_DEP_{l}" for l in LEAD_NAMES]
            + [f"T_INV_{l}" for l in LEAD_NAMES]
        )

        # Valid mask: all 51 tokens are valid sequence positions
        evidence_mask = torch.ones(B, evidence_tokens.size(1), dtype=torch.bool, device=device)

        # Expand candidate queries for batch
        query_tokens = self.candidate_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 8, d_model)

        return evidence_tokens, evidence_mask, evidence_token_names, query_tokens
