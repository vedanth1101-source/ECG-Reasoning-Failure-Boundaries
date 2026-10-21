"""
Dataset and DataLoader Utilities for Control Path Reasoning
===========================================================
Handles multimodal batch collation for:
    - Z_ECG (in R^{B x 768})
    - Structured clinical observations O_l (List of B dicts)
    - Lead availability mask m_l (in {0, 1}^{B x 12})
    - Multi-hot ground truth labels Y (in {0, 1}^{B x 8})

Provides factory methods bound directly to MultiCohortPartitionManager and
integrates deterministic rule-based observation extraction via ClinicalObservationExtractor.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from src.observations.clinical_extractor import ClinicalObservationExtractor


class ControlReasoningDataset(Dataset):
    """
    Multimodal Dataset for Control Path Reasoning.
    Supports in-memory tensor representations and structured clinical observation dictionaries.
    """

    def __init__(
        self,
        z_ecg: Union[np.ndarray, torch.Tensor],
        observations: List[Dict[str, Any]],
        lead_masks: Union[np.ndarray, torch.Tensor],
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        label_masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
        record_ids: Optional[List[str]] = None,
        variants: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        patient_ids: Optional[List[str]] = None,
    ):
        assert len(z_ecg) == len(observations) == len(lead_masks), (
            f"Mismatched dataset lengths: z_ecg={len(z_ecg)}, obs={len(observations)}, masks={len(lead_masks)}"
        )
        if labels is not None:
            assert len(labels) == len(z_ecg), f"Labels count {len(labels)} does not match records {len(z_ecg)}"
        if label_masks is not None:
            assert len(label_masks) == len(z_ecg), f"Label masks count {len(label_masks)} does not match records {len(z_ecg)}"

        if isinstance(z_ecg, np.ndarray):
            self.z_ecg = torch.from_numpy(z_ecg.astype(np.float32))
        else:
            self.z_ecg = z_ecg.float()

        self.observations = observations

        if isinstance(lead_masks, np.ndarray):
            self.lead_masks = torch.from_numpy(lead_masks.astype(np.float32))
        else:
            self.lead_masks = lead_masks.float()

        if labels is not None:
            if isinstance(labels, np.ndarray):
                self.labels = torch.from_numpy(labels.astype(np.float32))
            else:
                self.labels = labels.float()
        else:
            self.labels = None

        if label_masks is not None:
            if isinstance(label_masks, np.ndarray):
                self.label_masks = torch.from_numpy(label_masks.astype(np.float32))
            else:
                self.label_masks = label_masks.float()
        else:
            self.label_masks = None

        self.record_ids = record_ids
        self.variants = variants
        self.datasets = datasets
        self.patient_ids = patient_ids

    def __len__(self) -> int:
        return len(self.z_ecg)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "z_ecg": self.z_ecg[idx],
            "observation": self.observations[idx],
            "lead_mask": self.lead_masks[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        if self.label_masks is not None:
            item["label_mask"] = self.label_masks[idx]
        if self.record_ids is not None:
            item["record_id"] = self.record_ids[idx]
        if self.variants is not None:
            item["variant"] = self.variants[idx]
        if self.datasets is not None:
            item["dataset"] = self.datasets[idx]
        if self.patient_ids is not None:
            item["patient_id"] = self.patient_ids[idx]
        return item

    @classmethod
    def from_partition_manager(
        cls,
        partition_manager: Any,
        split: str,
        variant: str = "12-lead",
        observations: Optional[List[Dict[str, Any]]] = None,
        allow_test: bool = False,
    ) -> "ControlReasoningDataset":
        """
        Factory method constructing dataset directly from MultiCohortPartitionManager for a single variant.
        """
        reps, labels, label_masks, record_ids = partition_manager.load_split_representations(
            split=split,
            variant=variant,
            allow_test=allow_test,
        )
        lead_mask = partition_manager.get_lead_mask(record_ids[0], variant=variant)
        lead_masks = np.tile(lead_mask, (len(record_ids), 1))

        # Retrieve dataset and patient metadata
        datasets = [partition_manager.get_record_metadata(rid)["source_dataset"] for rid in record_ids]
        patient_ids = [str(partition_manager.get_record_metadata(rid)["source_patient_id"]) for rid in record_ids]
        variants = [variant] * len(record_ids)

        if observations is None:
            extractor = ClinicalObservationExtractor()
            obs_list = []
            for rid in record_ids:
                sig = partition_manager.get_signal(rid, variant=variant)
                obs = extractor.extract(sig, lead_mask).to_dict()
                obs_list.append(obs)
            observations = obs_list

        return cls(
            z_ecg=reps,
            observations=observations,
            lead_masks=lead_masks,
            labels=labels,
            label_masks=label_masks,
            record_ids=record_ids,
            variants=variants,
            datasets=datasets,
            patient_ids=patient_ids,
        )


class MultiConfigurationReasoningDataset(ControlReasoningDataset):
    """
    Multimodal Dataset pooling examples across all seven lead degradation configurations:
    12-lead, 10-lead, 8-lead, 6-lead, 4-lead, 2-lead, 1-lead.

    Invariants:
    1. Preserves authoritative record-level split assignment (train, validation).
    2. Every example carries its exact variant continuous latent Z_ECG and binary availability mask m_l.
    3. Exactly 7 views per canonical record (never counted as 7 independent patients).
    """

    @classmethod
    def from_partition_manager_all_configurations(
        cls,
        partition_manager: Any,
        split: str,
        variants: Optional[List[str]] = None,
        observations_by_variant: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        allow_test: bool = False,
    ) -> "MultiConfigurationReasoningDataset":
        from src.reasoning.multi_cohort_loader import LEAD_CONFIG_MASKS

        if variants is None:
            variants = list(LEAD_CONFIG_MASKS.keys())

        all_z = []
        all_obs = []
        all_masks = []
        all_labels = []
        all_label_masks = []
        all_rids = []
        all_variants = []
        all_datasets = []
        all_patients = []

        for v in variants:
            reps, lbls, l_masks, rids = partition_manager.load_split_representations(
                split=split,
                variant=v,
                allow_test=allow_test,
            )
            v_mask = LEAD_CONFIG_MASKS[v]
            v_masks = np.tile(v_mask, (len(rids), 1))

            if observations_by_variant and v in observations_by_variant:
                v_obs = observations_by_variant[v]
            else:
                extractor = ClinicalObservationExtractor()
                v_obs = []
                for rid in rids:
                    sig = partition_manager.get_signal(rid, variant=v)
                    obs = extractor.extract(sig, v_mask).to_dict()
                    v_obs.append(obs)

            v_datasets = [partition_manager.get_record_metadata(rid)["source_dataset"] for rid in rids]
            v_patients = [str(partition_manager.get_record_metadata(rid)["source_patient_id"]) for rid in rids]

            all_z.append(reps)
            all_obs.extend(v_obs)
            all_masks.append(v_masks)
            all_labels.append(lbls)
            all_label_masks.append(l_masks)
            all_rids.extend(rids)
            all_variants.extend([v] * len(rids))
            all_datasets.extend(v_datasets)
            all_patients.extend(v_patients)

        combined_z = np.concatenate(all_z, axis=0)
        combined_masks = np.concatenate(all_masks, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        combined_label_masks = np.concatenate(all_label_masks, axis=0)

        return cls(
            z_ecg=combined_z,
            observations=all_obs,
            lead_masks=combined_masks,
            labels=combined_labels,
            label_masks=combined_label_masks,
            record_ids=all_rids,
            variants=all_variants,
            datasets=all_datasets,
            patient_ids=all_patients,
        )


def collate_control_reasoning_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collator returning tensors and structured observation dictionaries.
    """
    z_ecg = torch.stack([b["z_ecg"] for b in batch])
    lead_masks = torch.stack([b["lead_mask"] for b in batch])
    observations = [b["observation"] for b in batch]

    collated = {
        "z_ecg": z_ecg,
        "observations": observations,
        "lead_masks": lead_masks,
    }

    if "label" in batch[0]:
        collated["labels"] = torch.stack([b["label"] for b in batch])

    if "label_mask" in batch[0]:
        collated["label_masks"] = torch.stack([b["label_mask"] for b in batch])

    if "record_id" in batch[0]:
        collated["record_ids"] = [b["record_id"] for b in batch]

    if "variant" in batch[0]:
        collated["variants"] = [b["variant"] for b in batch]

    if "dataset" in batch[0]:
        collated["datasets"] = [b["dataset"] for b in batch]

    if "patient_id" in batch[0]:
        collated["patient_ids"] = [b["patient_id"] for b in batch]

    return collated
