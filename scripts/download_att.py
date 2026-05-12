"""Download the AT&T (ORL) Database of Faces and verify layout."""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_ROOT = Path("data/att")
EXPECTED_SUBJECTS = 40
EXPECTED_PER_SUBJECT = 10

MIRRORS = [
    "https://github.com/Mongkok/att-database-of-faces/raw/main/att_faces.zip",
    "https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip",
]


def already_present() -> bool:
    if not DATA_ROOT.exists():
        return False
    for i in range(1, EXPECTED_SUBJECTS + 1):
        subject = DATA_ROOT / f"s{i:02d}"
        if not subject.is_dir():
            return False
        pgms = sorted(subject.glob("*.pgm"))
        if len(pgms) != EXPECTED_PER_SUBJECT:
            return False
    return True


def download_zip() -> bytes:
    last_error = None
    for url in MIRRORS:
        print(f"[download_att] trying {url}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            print(f"[download_att]   failed: {exc}", flush=True)
            last_error = exc
    raise RuntimeError(
        "All mirrors failed. Download att_faces.zip manually from "
        "https://www.kaggle.com/datasets/kasikrit/att-database-of-faces and "
        f"extract into {DATA_ROOT}. Last error: {last_error}"
    )


def extract_normalize(zip_bytes: bytes) -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if not parts[-1].endswith(".pgm") or len(parts) < 2:
                continue
            # zip uses unpadded "s1".."s40" but we normalize to "s01".."s40"
            subject_raw = parts[-2]
            if not (subject_raw.startswith("s") and subject_raw[1:].isdigit()):
                continue
            subject = f"s{int(subject_raw[1:]):02d}"
            target = DATA_ROOT / subject / parts[-1]
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())


def main() -> int:
    if already_present():
        print(f"[download_att] data already in {DATA_ROOT}, skipping.")
        return 0
    print("[download_att] starting download…")
    zip_bytes = download_zip()
    print(f"[download_att] got {len(zip_bytes)} bytes, extracting…")
    extract_normalize(zip_bytes)
    if not already_present():
        print(f"[download_att] ERROR: layout incorrect — inspect {DATA_ROOT}.")
        return 1
    print(f"[download_att] OK — {EXPECTED_SUBJECTS} subjects × {EXPECTED_PER_SUBJECT} PGMs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
