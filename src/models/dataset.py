"""
PTB-XL PyTorch Dataset and DataLoader Utilities (Phase 3)
=========================================================
Provides memory-mapped streaming from standardized signals.npy and labels.npy
with fold-stratified quarantine and on-the-fly lead masking augmentation.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.degradation.lead_masking import LeadMaskingOperator, LeadMask


class PTBXLECGDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL standardized 12-lead ECG signals.

    Args:
        data_dir: Path to directory containing signals.npy, labels.npy, and preprocessed_database.csv.
        split: 'train' (folds 1-8), 'val' (fold 9), or 'test' (fold 10).
        apply_lead_masking: If True, randomly applies Phase 2 lead degradation masks during training.
        mask_schedule: DegradationSchedule enum or name ('uniform' or 'anatomical').
        seed: Random seed for reproducible masking.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/processed_100hz",
        split: str = "train",
        apply_lead_masking: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split.lower()
        self.apply_lead_masking = apply_lead_masking
        self.rng = np.random.default_rng(seed)

        # 1. Load database metadata to identify fold assignments
        db_csv = self.data_dir / "preprocessed_database.csv"
        if not db_csv.exists():
            raise FileNotFoundError(f"Database metadata not found at {db_csv}")

        df = pd.read_csv(db_csv, low_memory=False)
        if "strat_fold" not in df.columns:
            raise KeyError("Column 'strat_fold' not found in database CSV.")

        # Determine split indices matching Phase 1 patient-stratified quarantine
        if self.split == "train":
            split_mask = df["strat_fold"].isin(range(1, 9))
        elif self.split in ("val", "validation"):
            split_mask = df["strat_fold"] == 9
        elif self.split == "test":
            split_mask = df["strat_fold"] == 10
        else:
            raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")

        self.indices = np.where(split_mask.values)[0]

        # 2. Memory-map signals to prevent RAM exhaustion (1000 samples, 12 channels)
        signals_path = self.data_dir / "signals.npy"
        labels_path = self.data_dir / "labels.npy"

        if not signals_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"signals.npy or labels.npy missing in {self.data_dir}")

        self.signals_mmap = np.load(signals_path, mmap_mode="r")
        self.labels_all = np.load(labels_path)

        # 3. Lead masking operator for dynamic augmentation
        self.mask_operator = LeadMaskingOperator()
        self.mask_names = ["12_lead", "10_lead", "8_lead", "6_lead", "4_lead", "2_lead", "1_lead"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.indices[idx]

        # Signal shape: (12, T), copy from mmap
        signal = np.array(self.signals_mmap[real_idx], dtype=np.float32)
        label = np.array(self.labels_all[real_idx], dtype=np.float32)

        # Default full 12-lead mask
        mask_vector = np.ones(12, dtype=np.float32)

        # Apply random lead masking during training if enabled
        if self.apply_lead_masking:
            mask_name = str(self.rng.choice(self.mask_names))
            masked_ecg = self.mask_operator.apply(signal, mask_name)
            signal = masked_ecg.signal
            mask_vector = masked_ecg.mask.mask_vector.astype(np.float32)

        return {
            "signal": torch.from_numpy(signal),            # (12, 1000)
            "label": torch.from_numpy(label),              # (8,)
            "mask": torch.from_numpy(mask_vector),          # (12,)
            "index": torch.tensor(real_idx, dtype=torch.long),
        }


def create_dataloader(
    data_dir: Union[str, Path] = "data/processed_100hz",
    split: str = "train",
    batch_size: int = 64,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    apply_lead_masking: bool = False,
    seed: int = 42,
) -> DataLoader:
    """
    Convenience factory function for creating PTB-XL PyTorch DataLoaders.
    """
    if shuffle is None:
        shuffle = (split.lower() == "train")

    dataset = PTBXLECGDataset(
        data_dir=data_dir,
        split=split,
        apply_lead_masking=apply_lead_masking,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split.lower() == "train"),
    )
