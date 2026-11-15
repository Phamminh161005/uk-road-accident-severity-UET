"""Download STATS19 raw files and reference documents from DfT.

Usage:
    python -m src.etl.download            # skip files that already exist
    python -m src.etl.download --force    # re-download everything
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from src.config import RAW_DIR, REFERENCE_DIR

STATS19_BASE = "https://data.dft.gov.uk/road-accidents-safety-data"
GOV_ASSETS = "https://assets.publishing.service.gov.uk/media"

# (url, target directory)
SOURCES: dict[str, tuple[str, Path]] = {
    "collision": (
        f"{STATS19_BASE}/dft-road-casualty-statistics-collision-last-5-years.csv",
        RAW_DIR,
    ),
    "vehicle": (
        f"{STATS19_BASE}/dft-road-casualty-statistics-vehicle-last-5-years.csv",
        RAW_DIR,
    ),
    "casualty": (
        f"{STATS19_BASE}/dft-road-casualty-statistics-casualty-last-5-years.csv",
        RAW_DIR,
    ),
    "data_guide": (
        f"{GOV_ASSETS}/6ab2a71d997a4b2950cced58/"
        "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx",
        REFERENCE_DIR,
    ),
    "severity_guidance": (
        f"{GOV_ASSETS}/691c644021ef5aaa6543eef0/"
        "dft-road-casualty-statistics-severity-adjustment-figure-guidance.docx",
        REFERENCE_DIR,
    ),
}


def download_file(url: str, target: Path, force: bool = False) -> Path:
    """Download ``url`` to ``target``; write to a temp file first so partial downloads never remain."""
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "stats19-research-client"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp.replace(target)
    return target


def download_all(force: bool = False) -> dict[str, Path]:
    """Download every file in ``SOURCES`` and return name -> local path."""
    paths = {}
    for name, (url, directory) in SOURCES.items():
        target = directory / url.rsplit("/", 1)[-1]
        paths[name] = download_file(url, target, force=force)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    for name, path in download_all(force=args.force).items():
        size_mb = path.stat().st_size / 1e6
        print(f"{name:<18} {size_mb:8.1f} MB  {path}")


if __name__ == "__main__":
    main()
