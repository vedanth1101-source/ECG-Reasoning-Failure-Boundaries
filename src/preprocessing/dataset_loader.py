"""
PTB-XL Dataset Loader and Cohort Manager
---------------------------------------
Provides high-throughput loading and standardized processing of the PTB-XL
benchmark dataset, respecting patient-stratified cross-validation splits.
"""

import os
import ast
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Generator
import numpy as np
import pandas as pd
import wfdb

from .pipeline import ECGPreprocessingPipeline, PreprocessedECG


class PTBXLDatasetLoader:
    """
    Loads and feeds PTB-XL 12-lead ECG records through the Phase 1 Preprocessing Pipeline.
    """

    def __init__(
        self,
        dataset_root: str,
        pipeline: Optional[ECGPreprocessingPipeline] = None,
        sampling_rate: int = 100,
    ):
        """
        Args:
            dataset_root: Path to directory containing ptbxl_database.csv and records100/records500.
            pipeline: Preprocessing pipeline instance (default creates target_fs=sampling_rate).
            sampling_rate: 100 or 500 Hz.
        """
        self.root = Path(dataset_root)
        self.sampling_rate = sampling_rate
        self.pipeline = pipeline or ECGPreprocessingPipeline(
            target_fs=float(sampling_rate),
            target_duration_sec=10.0
        )

        db_path = self.root / "ptbxl_database.csv"
        if not db_path.exists():
            raise FileNotFoundError(f"ptbxl_database.csv not found at {db_path}")

        # Load database metadata
        self.df = pd.read_csv(db_path, index_col="ecg_id")
        self.df["scp_codes_dict"] = self.df["scp_codes"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Load SCP statement aggregation if present
        scp_path = self.root / "scp_statements.csv"
        if scp_path.exists():
            self.scp_df = pd.read_csv(scp_path, index_col=0)
            self.diag_scp = self.scp_df[self.scp_df["diagnostic"] == 1]
            self.df["diagnostic_superclass"] = self.df["scp_codes_dict"].apply(self._aggregate_superclass)
        else:
            self.scp_df = None
            self.df["diagnostic_superclass"] = [[] for _ in range(len(self.df))]

    def _aggregate_superclass(self, scp_dict: Dict[str, float]) -> List[str]:
        if not isinstance(scp_dict, dict) or self.diag_scp is None:
            return []
        classes = []
        for code in scp_dict.keys():
            if code in self.diag_scp.index:
                diag_class = self.diag_scp.loc[code, "diagnostic_class"]
                if pd.notna(diag_class):
                    classes.append(str(diag_class))
        return sorted(list(set(classes)))

    def get_record_path(self, ecg_id: int) -> Path:
        """Resolves relative file path for an ecg_id."""
        row = self.df.loc[ecg_id]
        rel_path = row["filename_lr"] if self.sampling_rate == 100 else row["filename_hr"]
        return self.root / rel_path

    def load_record(self, ecg_id: int) -> PreprocessedECG:
        """
        Loads raw WFDB record and executes the 5-stage preprocessing pipeline.
        """
        file_path = str(self.get_record_path(ecg_id))
        raw_signal, meta = wfdb.rdsamp(file_path)

        lead_names = meta.get("sig_name", [])
        fs_in = float(meta.get("fs", self.sampling_rate))
        units = meta.get("units", "mV")
        gains = meta.get("adc_gain", None)
        baselines = meta.get("baseline", None)

        # Note: wfdb returns (T, C), pipeline handles transposition to (C, T)
        preprocessed = self.pipeline.process(
            raw_signal=raw_signal,
            lead_names=lead_names,
            fs_in=fs_in,
            adc_gain=gains,
            baseline=baselines,
            units=units
        )
        return preprocessed

    def iterate_cohort(
        self,
        strat_folds: Optional[List[int]] = None,
        limit: Optional[int] = None
    ) -> Generator[Tuple[int, PreprocessedECG, Dict[str, Any]], None, None]:
        """
        Generator yielding preprocessed records matching specified folds.
        """
        sub_df = self.df
        if strat_folds is not None:
            sub_df = sub_df[sub_df["strat_fold"].isin(strat_folds)]

        count = 0
        for ecg_id, row in sub_df.iterrows():
            if limit is not None and count >= limit:
                break
            try:
                record = self.load_record(ecg_id)
                meta = {
                    "patient_id": row.get("patient_id"),
                    "age": row.get("age"),
                    "sex": row.get("sex"),
                    "strat_fold": row.get("strat_fold"),
                    "diagnostic_superclass": row.get("diagnostic_superclass", []),
                }
                yield int(ecg_id), record, meta
                count += 1
            except Exception as e:
                # Log error or yield rejection
                continue
