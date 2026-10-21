"""
Control Path Reasoning Transformer (ECG-Only)
=============================================
Implementation of the shared Reasoning Transformer operating strictly in Control Mode.
Features:
    - Multimodal context ingestion (Z_ECG, O_l, lead mask, 8 candidate queries)
    - Transformer self-attention across physical evidence tokens
    - Cross-attention head mapping candidate hypotheses to observable evidence
    - Score Head producing 8 independent multi-label posterior probabilities P(H_i)
    - Attention Head producing normalized Evidence -> Hypothesis weight matrix A
    - Strict enforcement of Zero CKP tokens
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.reasoning.context_builder import ControlContextBuilder, CANDIDATE_ACRONYMS


class ControlReasoningTransformer(nn.Module):
    """
    Control Path Reasoning Transformer.
    Pits continuous ECG representations and discrete observations against
    candidate hypotheses without external ontology priors.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ecg: int = 768,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_candidates: int = 8,
        activation: str = "gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ecg = d_ecg
        self.num_candidates = num_candidates
        self.candidate_acronyms = CANDIDATE_ACRONYMS

        # 1. Context Builder
        self.context_builder = ControlContextBuilder(
            d_model=d_model,
            d_ecg=d_ecg,
            num_candidates=num_candidates,
            dropout=dropout,
        )

        # 2. Evidence Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.evidence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # 3. Cross-Attention: Hypotheses Querying Evidence
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm1 = nn.LayerNorm(d_model)
        self.query_norm2 = nn.LayerNorm(d_model)

        self.query_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # 4. Output Score Head (Independent sigmoid classification for 8 diseases)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        z_ecg: torch.Tensor,
        observations: List[Dict[str, Any]],
        lead_avail_mask: torch.Tensor,
        ckp_tokens: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Forward reasoning execution.

        Args:
            z_ecg: (B, 768) latent representation from FrozenECGFM
            observations: List of B structured observation dicts
            lead_avail_mask: (B, 12) binary mask
            ckp_tokens: MUST BE NONE in Control Path

        Returns:
            Dictionary containing:
                - "logits": (B, 8) classification logits
                - "probabilities": (B, 8) sigmoid posteriors P(H_i)
                - "attention_weights": (B, 8, N_evidence) normalized cross-attention weights
                - "evidence_names": List of token names corresponding to attention columns
                - "candidate_acronyms": List of 8 candidate acronyms
        """
        if ckp_tokens is not None:
            raise ValueError("[CONTROL PATH VIOLATION] Non-null CKP passed to ControlReasoningTransformer.")

        # Step 1: Context construction
        evidence_tokens, evidence_mask, evidence_names, query_tokens = self.context_builder(
            z_ecg=z_ecg,
            observations=observations,
            lead_avail_mask=lead_avail_mask,
            ckp_tokens=None,
        )

        # Step 2: Evidence self-attention encoding
        encoded_evidence = self.evidence_encoder(evidence_tokens)  # (B, N_ev, d_model)

        # Step 3: Cross-attention from candidate queries to evidence
        # Queries: (B, 8, d), Keys/Values: (B, N_ev, d)
        attn_out, attn_weights = self.cross_attention(
            query=query_tokens,
            key=encoded_evidence,
            value=encoded_evidence,
            need_weights=True,
            average_attn_weights=True,
        )  # attn_out: (B, 8, d), attn_weights: (B, 8, N_ev)

        # Step 4: Residual query update
        queries_updated = self.query_norm1(query_tokens + attn_out)
        queries_final = self.query_norm2(queries_updated + self.query_ffn(queries_updated))  # (B, 8, d)

        # Step 5: Score Head evaluation
        logits = self.score_head(queries_final).squeeze(-1)  # (B, 8)
        probabilities = torch.sigmoid(logits)                # (B, 8)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "attention_weights": attn_weights,
            "evidence_names": evidence_names,
            "candidate_acronyms": self.candidate_acronyms,
            "path_type": "CONTROL_ECG_ONLY",
        }
