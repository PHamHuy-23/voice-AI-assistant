import argparse
import os
import sys

# Ensure src package is discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_yaml
from src.data.audit import DatasetAuditor


def main():
    parser = argparse.ArgumentParser(description="Audit audio dataset distributions and generate report CSVs.")
    parser.add_argument("--config", type=str, default="configs/data/gsc.yaml", help="Path to dataset config YAML.")
    parser.add_argument("--data_dir", type=str, default=None, help="Override raw data directory.")
    parser.add_argument("--output_dir", type=str, default="reports/data_audit", help="Directory to save audit CSVs.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_dir = args.data_dir or cfg.get("data_dir", "data/raw/speech_commands_v2")
    file_ext = cfg.get("file_extension", ".wav")
    alias = cfg.get("alias", "dataset").lower()

    print(f"[*] Starting Dataset Audit on: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"[!] Warning: Data directory '{data_dir}' not found on local disk.")
        print(f"[!] If running on Kaggle, pass --data_dir /kaggle/input/speech-commands-v2")
        print("[!] Generating empty audit report template...")

    auditor = DatasetAuditor(data_dir=data_dir, file_ext=file_ext)
    df = auditor.run_full_audit(ignore_folders=cfg.get("special_folders", ["_background_noise_"]))

    out_file = auditor.export_reports(df, output_dir=args.output_dir, prefix=alias)
    print(f"[+] Audit completed. Summary saved to: {out_file}")
    if not df.empty:
        summary = auditor.generate_summary(df)
        print("\n--- DATASET AUDIT SUMMARY ---")
        for k, v in summary.items():
            if k != "classes":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
