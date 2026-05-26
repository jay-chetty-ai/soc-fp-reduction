"""Download CICIDS2017 dataset CSVs from the Canadian Institute for Cybersecurity.

The dataset is available at https://www.unb.ca/cic/datasets/ids-2017.html
Registration may be required. If automatic download fails, follow the manual
instructions printed to stdout.

Usage:
    python scripts/download_data.py [--raw-dir data/raw] [--fixture-dir data/fixtures]
"""

import argparse
import hashlib
import io
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Primary download source (UNB CIC). The URL redirects depending on their
# server configuration; we follow redirects up to 3 hops.
_PRIMARY_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip"

_MANUAL_INSTRUCTIONS = """
CICIDS2017 manual download instructions
========================================
The automatic download URL is no longer publicly accessible. Follow these steps:

1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Click "Download Dataset" and complete any registration form.
3. Download "MachineLearningCSV.zip" (approx. 360 MB).
4. Unzip the archive and place the CSV files in the directory:
       {raw_dir}
   Expected files:
       Monday-WorkingHours.pcap_ISCX.csv
       Tuesday-WorkingHours.pcap_ISCX.csv
       Wednesday-workingHours.pcap_ISCX.csv
       Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
       Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
       Friday-WorkingHours-Morning.pcap_ISCX.csv
       Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
       Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

5. After placing the CSVs, run this script again with --skip-download to
   create the 10K fixture:
       python scripts/download_data.py --skip-download

Alternatively, the dataset is mirrored on Kaggle:
   https://www.kaggle.com/datasets/cicdataset/cicids2017
   (requires a free Kaggle account and the kaggle CLI)
"""


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _attempt_download(url: str, dest: Path, timeout: int = 60) -> bool:
    """Try to download url to dest. Returns True on success, False on failure."""
    try:
        logger.info("Attempting download from %s", url)
        response = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
        content_type = response.headers.get("Content-Type", "")
        if response.status_code != 200 or "html" in content_type.lower():
            logger.warning(
                "Download failed: HTTP %s, Content-Type=%s",
                response.status_code,
                content_type,
            )
            return False
        with open(dest, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        logger.info("Downloaded %d bytes to %s", dest.stat().st_size, dest)
        return True
    except Exception as e:
        logger.warning("Download error: %s", e)
        return False


def _extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    """Extract all CSVs from a zip file into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                dest = out_dir / Path(name).name
                with zf.open(name) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                csv_paths.append(dest)
                logger.info("Extracted %s (%d bytes)", dest.name, dest.stat().st_size)
    return csv_paths


def _create_fixture(raw_dir: Path, fixture_dir: Path, config: dict) -> None:
    """Load the raw CSVs and write the 10K stratified fixture."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.loader import create_fixture_subset, load_dataset, validate_schema

    df = load_dataset(config)
    validate_schema(df)
    fixture = create_fixture_subset(df, n=config["data"]["fixture_size"])
    fixture_dir.mkdir(parents=True, exist_ok=True)
    dest = fixture_dir / "fixture_10k.csv"
    fixture.to_csv(dest, index=False)
    logger.info("Fixture saved to %s (%d rows).", dest, len(fixture))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare CICIDS2017.")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; assume CSVs are already in raw_dir.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["data"]["raw_dir"])
    fixture_dir = Path(config["data"]["fixtures_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        existing_csvs = list(raw_dir.glob("*.csv"))
        if existing_csvs:
            logger.info(
                "Found %d existing CSV(s) in %s; skipping download.",
                len(existing_csvs),
                raw_dir,
            )
        else:
            tmp_zip = raw_dir / "_cicids2017.zip"
            success = _attempt_download(_PRIMARY_URL, tmp_zip)
            if success:
                logger.info("Extracting archive...")
                extracted = _extract_zip(tmp_zip, raw_dir)
                tmp_zip.unlink(missing_ok=True)
                logger.info("Extracted %d CSV files.", len(extracted))
            else:
                print(_MANUAL_INSTRUCTIONS.format(raw_dir=raw_dir.resolve()))
                sys.exit(1)

    _create_fixture(raw_dir, fixture_dir, config)
    logger.info("Done. Run: pytest tests/test_epic1_data.py -v")


if __name__ == "__main__":
    main()
