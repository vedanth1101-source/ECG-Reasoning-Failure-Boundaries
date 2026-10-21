"""
Robust Downloader for Pretrained ECG-FM Model Weights
=====================================================
Downloads mimic_iv_ecg_physionet_pretrained.pt (~1.09 GB) directly via HTTP streaming
with resume support and integrity verification.
"""

import os
import sys
import time
from pathlib import Path
import requests

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

CHECKPOINT_URL = "https://huggingface.co/wanglab/ecg-fm/resolve/main/mimic_iv_ecg_physionet_pretrained.pt"
CONFIG_URL = "https://huggingface.co/wanglab/ecg-fm/resolve/main/mimic_iv_ecg_physionet_pretrained.yaml"
EXPECTED_SIZE = 1090825421  # 1.09 GB


def download_file(url: str, dest_path: Path, expected_size: int = None) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    initial_bytes = 0

    if temp_path.exists():
        initial_bytes = temp_path.stat().st_size
        if expected_size and initial_bytes == expected_size:
            print(f"[*] Partial download already complete. Finalizing {dest_path.name}...")
            temp_path.rename(dest_path)
            return
        headers["Range"] = f"bytes={initial_bytes}-"
        print(f"[*] Resuming download from byte {initial_bytes:,}...")

    print(f"[*] Connecting to {url}...")
    response = requests.get(url, headers=headers, stream=True, timeout=30)

    if response.status_code not in (200, 206):
        raise RuntimeError(f"HTTP Error {response.status_code} while downloading {url}")

    total_size = int(response.headers.get("content-length", 0)) + initial_bytes
    if expected_size and total_size != expected_size:
        print(f"[*] Note: Content length is {total_size:,} bytes.")

    chunk_size = 1024 * 1024  # 1 MB
    downloaded = initial_bytes
    start_time = time.time()
    last_print = start_time

    mode = "ab" if initial_bytes > 0 else "wb"
    with open(temp_path, mode) as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print >= 5.0 or downloaded >= total_size:
                    elapsed = now - start_time
                    speed_mb = (downloaded - initial_bytes) / (1024 * 1024 * max(elapsed, 0.001))
                    pct = (downloaded / total_size * 100) if total_size > 0 else 0
                    print(
                        f"  --> Progress: {downloaded / (1024*1024):.1f} MB / "
                        f"{total_size / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed_mb:.2f} MB/s"
                    )
                    last_print = now

    # Finalize
    if dest_path.exists():
        dest_path.unlink()
    temp_path.rename(dest_path)
    print(f"[+] Successfully downloaded {dest_path.name} ({dest_path.stat().st_size:,} bytes).")


def main():
    target_dir = Path("checkpoints/ecg_fm")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download yaml config
    yaml_dest = target_dir / "mimic_iv_ecg_physionet_pretrained.yaml"
    if not yaml_dest.exists():
        print(f"[*] Downloading {yaml_dest.name}...")
        download_file(CONFIG_URL, yaml_dest)
    else:
        print(f"[*] Config {yaml_dest.name} already exists.")

    # 2. Download model checkpoint
    ckpt_dest = target_dir / "mimic_iv_ecg_physionet_pretrained.pt"
    if not ckpt_dest.exists() or ckpt_dest.stat().st_size != EXPECTED_SIZE:
        print(f"[*] Downloading model checkpoint {ckpt_dest.name}...")
        download_file(CHECKPOINT_URL, ckpt_dest, expected_size=EXPECTED_SIZE)
    else:
        print(f"[+] Checkpoint {ckpt_dest.name} already present and verified ({ckpt_dest.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
