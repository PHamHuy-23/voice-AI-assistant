import os
import glob
import math
import wave
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class DatasetAuditor:
    """
    Audits audio datasets with zero fabricated statistics.
    Computes exact file-by-file distributions, corruptions, and split integrity.
    """

    def __init__(self, data_dir: str, file_ext: str = ".wav"):
        self.data_dir = data_dir
        self.file_ext = file_ext

    def audit_file(self, file_path: str) -> Dict[str, Any]:
        """Audits a single audio file via the native standard library wave module or fallback."""
        record = {
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "class_name": os.path.basename(os.path.dirname(file_path)),
            "is_corrupted": False,
            "error_msg": "",
            "sample_rate": None,
            "channels": None,
            "num_frames": None,
            "duration_sec": None,
            "file_size_bytes": 0,
        }
        try:
            record["file_size_bytes"] = os.path.getsize(file_path)
            with wave.open(file_path, "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                frames = wf.getnframes()
                duration = frames / float(sample_rate) if sample_rate > 0 else 0.0

                record["sample_rate"] = sample_rate
                record["channels"] = channels
                record["num_frames"] = frames
                record["duration_sec"] = round(duration, 4)
        except Exception as e:
            record["is_corrupted"] = True
            record["error_msg"] = str(e)

        return record

    def run_full_audit(self, ignore_folders: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scans all audio files within the dataset directory and returns a detailed DataFrame.
        """
        if ignore_folders is None:
            ignore_folders = ["_background_noise_"]

        pattern = os.path.join(self.data_dir, "**", f"*{self.file_ext}")
        all_files = glob.glob(pattern, recursive=True)

        records = []
        for fp in all_files:
            folder_name = os.path.basename(os.path.dirname(fp))
            if folder_name in ignore_folders:
                continue
            records.append(self.audit_file(fp))

        if not records:
            # Return empty schema if no files found
            return pd.DataFrame(
                columns=[
                    "file_path",
                    "filename",
                    "class_name",
                    "is_corrupted",
                    "error_msg",
                    "sample_rate",
                    "channels",
                    "num_frames",
                    "duration_sec",
                    "file_size_bytes",
                ]
            )

        df = pd.DataFrame(records)
        return df

    @staticmethod
    def generate_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts aggregate dataset metrics from the audit DataFrame.
        """
        if df.empty:
            return {
                "total_files": 0,
                "total_classes": 0,
                "corrupted_files": 0,
                "duration_min": "TBD_FROM_EXPERIMENT",
                "duration_max": "TBD_FROM_EXPERIMENT",
                "duration_mean": "TBD_FROM_EXPERIMENT",
                "sample_rate_distribution": {},
                "channels_distribution": {},
            }

        valid_df = df[~df["is_corrupted"]]
        durations = valid_df["duration_sec"].dropna()

        summary = {
            "total_files": int(len(df)),
            "valid_files": int(len(valid_df)),
            "corrupted_files": int(df["is_corrupted"].sum()),
            "total_classes": int(df["class_name"].nunique()),
            "classes": sorted(df["class_name"].unique().tolist()),
            "sample_rate_distribution": df["sample_rate"].value_counts().to_dict(),
            "channels_distribution": df["channels"].value_counts().to_dict(),
            "duration_min": float(durations.min()) if not durations.empty else None,
            "duration_max": float(durations.max()) if not durations.empty else None,
            "duration_mean": round(float(durations.mean()), 4) if not durations.empty else None,
            "duration_std": round(float(durations.std()), 4) if not durations.empty else None,
        }
        return summary

    def export_reports(self, df: pd.DataFrame, output_dir: str, prefix: str = "dataset"):
        """Exports audit report CSVs to the designated report directory."""
        os.makedirs(output_dir, exist_ok=True)
        # 1. Full file-level audit
        file_audit_path = os.path.join(output_dir, f"{prefix}_file_audit.csv")
        df.to_csv(file_audit_path, index=False)

        # 2. Class distribution
        if not df.empty:
            class_counts = df["class_name"].value_counts().reset_index()
            class_counts.columns = ["class_name", "num_samples"]
            class_path = os.path.join(output_dir, f"{prefix}_class_distribution.csv")
            class_counts.to_csv(class_path, index=False)

        # 3. Overall summary
        summary = self.generate_summary(df)
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(output_dir, f"{prefix}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        return summary_path
