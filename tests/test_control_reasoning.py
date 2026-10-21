"""
Unit Tests for Phase 4 Control Path (ECG-Only)
==============================================
Validates:
    - Multimodal context construction without CKP
    - Strict rejection of external CKP tokens (control isolation)
    - ControlReasoningTransformer forward pass, score head, and attention head
    - Structured reasoning trace format adherence
    - Observability flag alignment with lead availability mask
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.reasoning.context_builder import (
    ControlContextBuilder,
    LEAD_NAMES,
    CANDIDATE_ACRONYMS,
)
from src.reasoning.reasoning_transformer import ControlReasoningTransformer
from src.reasoning.trace_generator import StructuredTraceGenerator
from src.reasoning.dataset import (
    ControlReasoningDataset,
    collate_control_reasoning_batch,
)


@pytest.fixture
def sample_observations():
    return [
        {
            "rhythm": "Sinus rhythm",
            "heart_rate": 72,
            "pr_interval_ms": 160,
            "qrs_duration_ms": 90,
            "st_elevation": {
                "present": True,
                "leads": ["II", "III", "aVF"],
                "amplitude_mm": {"II": 1.5, "III": 1.8, "aVF": 1.2},
            },
            "st_depression": {
                "present": True,
                "leads": ["I", "aVL"],
                "amplitude_mm": {"I": -0.8, "aVL": -0.6},
            },
            "t_wave_inversion": {"present": False, "leads": []},
        },
        {
            "rhythm": "Atrial fibrillation",
            "heart_rate": 115,
            "pr_interval_ms": None,
            "qrs_duration_ms": 95,
            "st_elevation": {"present": False, "leads": [], "amplitude_mm": {}},
            "st_depression": {"present": False, "leads": [], "amplitude_mm": {}},
            "t_wave_inversion": {"present": True, "leads": ["V5", "V6"]},
        },
    ]


@pytest.fixture
def sample_lead_mask():
    # Record 0: full 12 leads (all 1)
    # Record 1: 4 leads active (I, II, III, V1) -> indices 0, 1, 2, 6
    mask = torch.zeros(2, 12, dtype=torch.float32)
    mask[0, :] = 1.0
    mask[1, [0, 1, 2, 6]] = 1.0
    return mask


@pytest.fixture
def sample_z_ecg():
    return torch.randn(2, 768)


class TestControlContextBuilder:
    def test_output_shapes_and_types(self, sample_z_ecg, sample_observations, sample_lead_mask):
        builder = ControlContextBuilder(d_model=128, d_ecg=768, num_candidates=8)
        ev_tokens, ev_mask, ev_names, q_tokens = builder(
            z_ecg=sample_z_ecg,
            observations=sample_observations,
            lead_avail_mask=sample_lead_mask,
        )

        assert ev_tokens.shape == (2, 51, 128)
        assert ev_mask.shape == (2, 51)
        assert q_tokens.shape == (2, 8, 128)
        assert len(ev_names) == 51
        assert ev_names[0] == "ECG_CONTINUOUS_FM"

    def test_ckp_tokens_strict_rejection(self, sample_z_ecg, sample_observations, sample_lead_mask):
        builder = ControlContextBuilder()
        with pytest.raises(ValueError, match="CONTROL PATH VIOLATION"):
            builder(
                z_ecg=sample_z_ecg,
                observations=sample_observations,
                lead_avail_mask=sample_lead_mask,
                ckp_tokens={"forbidden_prior": "Inferior MI requires ST elevation in II, III, aVF"},
            )

    def test_trained_linear_extractor_context_compatibility(self, sample_z_ecg, sample_lead_mask):
        from src.reasoning.observation_extractor import LinearObservationExtractor
        extractor = LinearObservationExtractor(d_in=768)
        # 1. Test taxonomy format (predict_discretized default)
        taxonomy_obs = extractor.predict_discretized(sample_z_ecg, sample_lead_mask, compatible_format=False)
        builder = ControlContextBuilder(d_model=128, d_ecg=768, num_candidates=8)
        ev_tokens, ev_mask, ev_names, q_tokens = builder(
            z_ecg=sample_z_ecg,
            observations=taxonomy_obs,
            lead_avail_mask=sample_lead_mask,
        )
        assert ev_tokens.shape == (2, 51, 128)
        assert q_tokens.shape == (2, 8, 128)

        # 2. Test compatible format (predict_discretized with compatible_format=True)
        compatible_obs = extractor.predict_discretized(sample_z_ecg, sample_lead_mask, compatible_format=True)
        ev_tokens2, _, _, _ = builder(
            z_ecg=sample_z_ecg,
            observations=compatible_obs,
            lead_avail_mask=sample_lead_mask,
        )
        assert ev_tokens2.shape == (2, 51, 128)


class TestControlReasoningTransformer:
    def test_forward_pass_and_heads(self, sample_z_ecg, sample_observations, sample_lead_mask):
        model = ControlReasoningTransformer(
            d_model=128,
            d_ecg=768,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            num_candidates=8,
        )
        model.eval()

        with torch.no_grad():
            output = model(
                z_ecg=sample_z_ecg,
                observations=sample_observations,
                lead_avail_mask=sample_lead_mask,
            )

        assert "logits" in output
        assert "probabilities" in output
        assert "attention_weights" in output
        assert "evidence_names" in output
        assert output["path_type"] == "CONTROL_ECG_ONLY"

        logits = output["logits"]
        probs = output["probabilities"]
        attn = output["attention_weights"]

        # 8 candidate predictions
        assert logits.shape == (2, 8)
        assert probs.shape == (2, 8)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

        # Attention matrix: (B, 8, 51)
        assert attn.shape == (2, 8, 51)
        # Check attention weights sum to 1.0 along evidence dimension
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_ckp_isolation_in_transformer(self, sample_z_ecg, sample_observations, sample_lead_mask):
        model = ControlReasoningTransformer(d_model=64, d_ecg=768, nhead=2, num_layers=1)
        with pytest.raises(ValueError, match="CONTROL PATH VIOLATION"):
            model(
                z_ecg=sample_z_ecg,
                observations=sample_observations,
                lead_avail_mask=sample_lead_mask,
                ckp_tokens={"concepts": ["MI"]},
            )


class TestStructuredTraceGenerator:
    def test_trace_generation_and_unobservable_leads(
        self, sample_z_ecg, sample_observations, sample_lead_mask
    ):
        model = ControlReasoningTransformer(d_model=64, d_ecg=768, nhead=2, num_layers=1)
        model.eval()

        with torch.no_grad():
            output = model(
                z_ecg=sample_z_ecg,
                observations=sample_observations,
                lead_avail_mask=sample_lead_mask,
            )

        generator = StructuredTraceGenerator(evidence_weight_threshold=0.01, top_evidence_k=5)
        traces = generator.generate_batch_traces(
            probabilities=output["probabilities"],
            attention_matrix=output["attention_weights"],
            evidence_names=output["evidence_names"],
            lead_avail_masks=sample_lead_mask,
            observations=sample_observations,
            record_ids=["ptb_0001", "ptb_0002"],
        )

        assert len(traces) == 2
        for trace in traces:
            assert trace["path_type"] == "CONTROL_ECG_ONLY"
            assert "top_hypothesis" in trace
            assert "acronym" in trace["top_hypothesis"]
            assert "confidence" in trace["top_hypothesis"]
            assert len(trace["posterior_probabilities"]) == 8
            assert "lead_observability" in trace
            assert isinstance(trace["supporting_evidence"], list)

        # Check second record (4 active leads: I, II, III, V1)
        trace_masked = traces[1]
        active_leads = set(trace_masked["lead_observability"]["active_leads"])
        assert active_leads == {"I", "II", "III", "V1"}

        # If any supporting evidence item references a masked lead, verify is_observable_in_mask is False
        for ev in trace_masked["supporting_evidence"]:
            assoc = ev["associated_leads"]
            if assoc:
                expected_observable = all(l in active_leads for l in assoc)
                assert ev["is_observable_in_mask"] == expected_observable


class TestControlReasoningDataset:
    def test_dataset_and_dataloader_collate(self, sample_z_ecg, sample_observations, sample_lead_mask):
        labels = torch.zeros(2, 8)
        labels[0, 7] = 1.0  # MI
        labels[1, 1] = 1.0  # AFIB

        dataset = ControlReasoningDataset(
            z_ecg=sample_z_ecg,
            observations=sample_observations,
            lead_masks=sample_lead_mask,
            labels=labels,
            record_ids=["rec_0", "rec_1"],
        )

        assert len(dataset) == 2
        item = dataset[0]
        assert "z_ecg" in item
        assert "observation" in item
        assert "lead_mask" in item
        assert "label" in item

        batch = [dataset[0], dataset[1]]
        collated = collate_control_reasoning_batch(batch)
        assert collated["z_ecg"].shape == (2, 768)
        assert collated["lead_masks"].shape == (2, 12)
        assert collated["labels"].shape == (2, 8)
        assert len(collated["observations"]) == 2
