#!/usr/bin/env python3
"""
Script 1: Download EMSR Flood Activations → standardized format

Pipeline steps (all in one script, resumable):
  1. Fetch all flood activations in date range from Copernicus API
  2. Download raw vectors + PDFs for each flood activation
  3. Find PDF per product, extract acquisition date
  4. Extract data sources (pre/post-event sensor info) from PDF → CSV
  5. Validate required shapefiles (aoi, event)
  6. Convert to standardized format (flat folders: aoi/, flood_extent/)

Resume logic:
  - data/metadata/activations_status.csv tracks per-product state
  - Already-completed products are skipped at every step
  - Safe to re-run at any time

Output structure (relative to BASE_DIR, auto-detected from script location):
  data/
  ├── activations/
  │   ├── activations_raw/                       ← raw downloads
  │   │   └── EMSR657/
  │   │       └── AOI01_DaugavaRiver_DEL_MONIT01/   ← product folder (vectors + PDFs)
  │   └── activations_reorganized/                       ← standardized format
  │       └── EMSR657/                          ← activation parent folder
  │           └── EMSR657_AOI01_DaugavaRiver_DEL_MONIT01_20230402/
  │               ├── aoi/aoi.shp          (+ .shx .dbf .prj ...)
  │               └── flood_extent/event.shp
  └── data/
      ├── activations_status.csv  ← internal per-product processing state
      ├── activations_sources.csv ← internal pre/post-event sensor info from PDFs
      └── activations.csv         ← first-draft catalog: folder, date, sensors

Skipped silently (not counted as failures):
  - Products with no PDF (not a delineation/monitoring map product)
  - Products with no event shapefile (reference/overview products)

Dependencies: requests, beautifulsoup4, PyMuPDF (fitz), urllib3
"""

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATE_START = "2026-02-10"   # flood event date range – start (inclusive, YYYY-MM-DD)
DATE_END   = "2026-02-19"   # flood event date range – end   (inclusive, YYYY-MM-DD)

# Activations >= this number use the newer dashboard API for product listing.
# Older ones are scraped from the HTML activation page.
API_THRESHOLD = 656

# Seconds to wait between web requests (be polite to Copernicus servers)
REQUEST_DELAY = 1.5

# Shapefile component extensions to copy into standardized folders
SHAPEFILE_EXTENSIONS = ['.shp', '.shx', '.dbf', '.prj', '.cpg',
                         '.sbn', '.sbx', '.shp.xml', '.xml']
# ─────────────────────────────────────────────────────────────────────────────

import csv
import json
import re
import shutil
import sys
import time
import urllib3
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


# ─── LOGGING SETUP ───────────────────────────────────────────────────────────

class _Tee:
    """Mirror stdout to a log file simultaneously."""
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        sys.stdout = self._stdout
        self._file.close()

def _setup_logging() -> _Tee:
    logs_dir = Path(__file__).resolve().parent / "logs"
    log_path  = logs_dir / f"1_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"Log: {log_path}")
    return tee

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed.  Run:  pip install PyMuPDF")
    sys.exit(1)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─── PATH SETUP ──────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent.parent   # repo root
DATA_DIR    = BASE_DIR / "data"
META_DIR    = DATA_DIR / "metadata"
ACTIVATIONS = DATA_DIR / "activations"
RAW_DIR     = ACTIVATIONS / "activations_raw"
ACT_DIR     = ACTIVATIONS / "activations_reorganized"
TEMP_DIR    = DATA_DIR / "_temp"

import config
STATUS_CSV  = config.CSV_ACTIVATION_STATUS    # 1_activation_status.csv (resume state)
CATALOG_CSV = config.CSV_ACTIVATION_CATALOG   # 1_activation_catalog.csv (output catalog)


# ─── STATUS TRACKER ──────────────────────────────────────────────────────────

STATUS_FIELDS = [
    "emsr_code", "product_folder", "event_type",
    "raw_downloaded", "pdf_found", "date_extracted", "event_date",
    "pre_event_sensor", "post_event_sensors",
    "act_folder", "reorg_converted", "has_aoi", "has_event",
    "skipped_reason", "last_updated",
]

class StatusTracker:
    """
    Reads and writes activations_status.csv.
    Key = (emsr_code, product_folder).
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._data: Dict[Tuple[str, str], Dict] = {}
        self._load()

    def _load(self):
        if not self.csv_path.exists():
            return
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["emsr_code"], row["product_folder"])
                self._data[key] = row

    def get(self, emsr_code: str, product_folder: str) -> Optional[Dict]:
        return self._data.get((emsr_code, product_folder))

    def upsert(self, record: Dict):
        key = (record["emsr_code"], record["product_folder"])
        record["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if key in self._data:
            self._data[key].update(record)
        else:
            # Fill missing fields with empty string
            full = {f: "" for f in STATUS_FIELDS}
            full.update(record)
            self._data[key] = full
        self._flush()

    def _flush(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
            writer.writeheader()
            for row in self._data.values():
                writer.writerow(row)

    def is_reorganized(self, emsr_code: str, product_folder: str) -> bool:
        rec = self.get(emsr_code, product_folder)
        return bool(rec and rec.get("reorg_converted") == "yes")

    def is_raw_done(self, emsr_code: str, product_folder: str) -> bool:
        rec = self.get(emsr_code, product_folder)
        return bool(rec and rec.get("raw_downloaded") == "yes")

    def is_non_flood(self, emsr_code: str) -> bool:
        """True if any record for this EMSR code marks it as non-flood."""
        for (code, _), rec in self._data.items():
            if code == emsr_code and rec.get("is_flood") == "no":
                return True
        return False


# ─── DATA SOURCES CSV ────────────────────────────────────────────────────────

SOURCES_FIELDS = [
    "activation_folder", "emsr_code", "product_folder",
    "pdf_file",
    "pre_event_sensor", "pre_event_date", "pre_event_gsd_m", "pre_event_cloud_pct",
    "post_event_sensor",  "post_event_date",  "post_event_gsd_m",  "post_event_cloud_pct",
    "post_event_sensor2", "post_event_date2", "post_event_gsd2",   "post_event_cloud2",
    "raw_sources_text",
]

class SourcesTracker:
    """Reads and writes activations_sources.csv."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._records: Dict[Tuple[str, str], Dict] = {}
        self._load()

    def _load(self):
        if not self.csv_path.exists():
            return
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["emsr_code"], row["product_folder"])
                self._records[key] = row

    def already_done(self, emsr_code: str, product_folder: str) -> bool:
        return (emsr_code, product_folder) in self._records

    def get_by_key(self, emsr_code: str, product_folder: str) -> Optional[Dict]:
        return self._records.get((emsr_code, product_folder))

    def append(self, record: Dict):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SOURCES_FIELDS,
                                    extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(record)
        key = (record["emsr_code"], record["product_folder"])
        self._records[key] = record


# ─── FLOOD FILTER ────────────────────────────────────────────────────────────

class FloodFilter:
    """
    Determines whether an EMSR code is a flood activation.
    Tries three methods in order:
      1. New dashboard API  (EMSR656+)
      2. Old mapping API    (all codes, works up to ~EMSR763)
      3. HTML page scraping (last resort)
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Referer":    "https://rapidmapping.emergency.copernicus.eu/",
        })

    def get_event_type(self, code: str) -> Optional[str]:
        num = _emsr_num(code)
        if num is None:
            return None

        # Method 1: new dashboard API (EMSR656+)
        if num >= API_THRESHOLD:
            url = (
                "https://rapidmapping.emergency.copernicus.eu/"
                f"backend/dashboard-api/public-activations-info/?code={code}"
            )
            try:
                r = self.session.get(url, timeout=15, verify=False)
                if r.status_code == 200:
                    data = r.json()
                    results = data.get("results", [])
                    if results:
                        return results[0].get("category")
            except Exception:
                pass

        # Method 2: old mapping API (works for all older codes)
        url = f"https://mapping.emergency.copernicus.eu/activations/api/activations/{code}/"
        try:
            r = self.session.get(url, timeout=15, verify=True)
            if r.status_code == 200:
                data = r.json()
                cat = data.get("category", {})
                if isinstance(cat, dict):
                    return cat.get("name")
                if isinstance(cat, str):
                    return cat
            elif r.status_code == 404:
                return None
        except Exception:
            pass

        # Method 3: HTML scraping
        url = f"https://mapping.emergency.copernicus.eu/activations/{code}"
        try:
            r = self.session.get(url, timeout=15, verify=True)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            text = soup.get_text()
            lines = [l.strip() for l in text.split("\n")]
            for i, line in enumerate(lines):
                if line.lower() == "event type" and i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    if nxt and nxt.lower() != "event type":
                        return nxt
            m = re.search(r"Event\s+type\s*:\s*(\w[\w\s]*)", text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        except Exception:
            pass

        return None

    def is_flood(self, code: str) -> Tuple[bool, Optional[str]]:
        """Returns (is_flood, event_type_string)."""
        event_type = self.get_event_type(code)
        if event_type is None:
            return False, None
        flood = bool(re.search(r"flood|inundation", event_type, re.IGNORECASE))
        return flood, event_type


# ─── DOWNLOADER ──────────────────────────────────────────────────────────────

class EMSRDownloader:
    """
    Downloads raw vector data (ZIP) and PDFs for one EMSR activation.

    API format  (EMSR >= API_THRESHOLD):
      - Uses dashboard API to list products → downloadPath (ZIP)
      - Extracts ZIP → activations_raw/EMSR{N}/{product_folder}/
      - PDF is found recursively inside the extracted folder

    HTML format (EMSR < API_THRESHOLD):
      - Scrapes the activation HTML page for PDF and vector.zip links
      - Extracts ZIP → activations_raw/EMSR{N}/{product_folder}/VECTOR/
      - PDF is downloaded directly into the product folder
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Referer":    "https://rapidmapping.emergency.copernicus.eu/",
        })

    # ── internal helpers ──────────────────────────────────────────────────

    def _download_file(self, url: str, dest: Path) -> bool:
        """Stream-download url → dest.  Returns True on success."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = self.session.get(url, stream=True, timeout=60, verify=False)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"        ✗ Download failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def _extract_zip(self, zip_path: Path, dest: Path) -> bool:
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dest)
            return True
        except Exception as e:
            print(f"        ✗ Extract failed: {e}")
            return False

    # ── API method (EMSR656+) ─────────────────────────────────────────────

    def _get_products_api(self, code: str) -> List[Dict]:
        """
        Query dashboard API → list of product dicts:
          {folder_name, aoi_num, aoi_name, product_label, vector_url, delivery_date}
        """
        api_url = (
            "https://rapidmapping.emergency.copernicus.eu/"
            f"backend/dashboard-api/public-activations/?code={code}"
        )
        try:
            r = self.session.get(api_url, timeout=30, verify=False)
            if r.status_code == 404:
                return []
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if not results:
                return []
            activation = results[0]
        except Exception as e:
            print(f"      API error for {code}: {e}")
            return []

        products = []
        for aoi in activation.get("aois", []):
            aoi_num  = f"AOI{aoi['number']:02d}"
            aoi_name = re.sub(r"[^\w]", "_", aoi.get("name", "")).strip("_")

            for prod in aoi.get("products", []):
                vector_url = prod.get("downloadPath", "")
                if not vector_url:
                    continue

                prod_type   = prod.get("type", "UNKNOWN")
                monit       = prod.get("monitoring", False)
                monit_num   = prod.get("monitoringNumber", 0)

                if monit and monit_num > 0:
                    label = f"{prod_type}_MONIT{monit_num:02d}"
                else:
                    label = f"{prod_type}_PRODUCT"

                # Delivery date from version info
                delivery_date = None
                ver = prod.get("version", {})
                if ver and "deliveryTime" in ver:
                    try:
                        dt = datetime.fromisoformat(
                            ver["deliveryTime"].replace("Z", "+00:00")
                        )
                        delivery_date = dt.strftime("%Y%m%d")
                    except Exception:
                        pass

                folder_name = f"{aoi_num}_{aoi_name}_{label}"

                products.append({
                    "folder_name":   folder_name,
                    "aoi_num":       aoi_num,
                    "aoi_name":      aoi_name,
                    "product_label": label,
                    "vector_url":    vector_url,
                    "pdf_url":       None,   # found later from extracted files
                    "delivery_date": delivery_date,
                })
        return products

    def _download_api_product(self, code: str, product: Dict,
                               activation_raw_dir: Path) -> bool:
        """Download and extract a single API-format product ZIP."""
        folder_name = product["folder_name"]
        dest_dir    = activation_raw_dir / folder_name

        if dest_dir.exists() and any(dest_dir.rglob("*.shp")):
            print(f"        ↩  already downloaded")
            return True

        zip_url  = product["vector_url"]
        zip_name = zip_url.split("/")[-1].split("?")[0] or f"{folder_name}.zip"
        zip_path = TEMP_DIR / zip_name

        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"        ↓  {zip_name}")

        if not self._download_file(zip_url, zip_path):
            return False

        dest_dir.mkdir(parents=True, exist_ok=True)
        ok = self._extract_zip(zip_path, dest_dir)
        zip_path.unlink(missing_ok=True)
        return ok

    # ── HTML method (EMSR < API_THRESHOLD) ───────────────────────────────

    def _get_products_html(self, code: str) -> List[Dict]:
        """
        Scrape the Copernicus HTML page for an older activation.
        Returns list of product dicts with pdf_url and vector_url.
        """
        page_url = f"https://mapping.emergency.copernicus.eu/activations/{code}"
        try:
            r = self.session.get(page_url, timeout=30, verify=True)
            if r.status_code == 404:
                return []
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
        except Exception as e:
            print(f"      HTML fetch error for {code}: {e}")
            return []

        products = []
        current_aoi_num  = None
        current_aoi_name = ""

        elements = soup.find_all(["p", "h5"])
        for elem in elements:
            if elem.name == "p":
                text = elem.get_text(strip=True)
                m = re.search(r"(AOI\d+)\s+(.+?)(?:\s*[-–]\s*|\s*$)", text)
                if m:
                    current_aoi_num  = m.group(1)
                    current_aoi_name = re.sub(r"[^\w]", "_", m.group(2).strip()).strip("_")

            elif elem.name == "h5" and current_aoi_num:
                product_name = elem.get_text(strip=True)
                if not product_name or len(product_name) < 3:
                    continue

                # Delivery date from sibling text
                delivery_date = None
                parent = elem.find_parent()
                if parent:
                    parent_text = parent.get_text()
                    dm = re.search(r"Delivery:\s*(\d{4}-\d{2}-\d{2})", parent_text)
                    if dm:
                        delivery_date = dm.group(1).replace("-", "")

                # Find PDF and vector.zip links near this h5
                pdf_url    = None
                vector_url = None
                if parent:
                    for a in parent.find_all("a", href=True):
                        href = a["href"]
                        full = href if href.startswith("http") else urljoin(page_url, href)
                        if href.lower().endswith(".pdf"):
                            pdf_url = full
                        elif "vector.zip" in href.lower():
                            vector_url = full

                if not pdf_url and not vector_url:
                    continue

                clean_pname = re.sub(r"[^\w]", "_", product_name).strip("_")
                folder_name = f"{current_aoi_num}_{current_aoi_name}_{clean_pname}"

                products.append({
                    "folder_name":   folder_name,
                    "aoi_num":       current_aoi_num,
                    "aoi_name":      current_aoi_name,
                    "product_label": clean_pname,
                    "vector_url":    vector_url,
                    "pdf_url":       pdf_url,
                    "delivery_date": delivery_date,
                })

        # Fallback: scrape all links if AOI parsing found nothing
        if not products:
            products = self._get_products_html_fallback(soup, page_url, code)

        return products

    def _get_products_html_fallback(self, soup, page_url: str, code: str) -> List[Dict]:
        """
        Fallback for old activations whose HTML doesn't have the AOI/h5 structure.
        Groups PDF and vector.zip links by the product identifier found in the filename.
        """
        product_map: Dict[str, Dict] = {}

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if code not in href:
                continue
            if not (href.lower().endswith(".pdf") or "vector.zip" in href.lower()):
                continue

            filename = href.split("/")[-1].split("?")[0]
            # Try to parse: EMSR018_01NAME_DELINEATION_DETAIL01_v1_...
            m = re.match(r"(EMSR\d+)_([^_]+)_(.+?)_(?:r\d+|v\d+)", filename)
            if not m:
                continue
            _, aoi_part, product_part = m.groups()

            key = f"{aoi_part}_{product_part}"
            if key not in product_map:
                product_map[key] = {
                    "folder_name":   key,
                    "aoi_num":       aoi_part,
                    "aoi_name":      "",
                    "product_label": product_part,
                    "vector_url":    None,
                    "pdf_url":       None,
                    "delivery_date": None,
                }

            full = href if href.startswith("http") else urljoin(page_url, href)
            if href.lower().endswith(".pdf"):
                product_map[key]["pdf_url"] = full
            else:
                product_map[key]["vector_url"] = full

        return list(product_map.values())

    def _download_html_product(self, code: str, product: Dict,
                                activation_raw_dir: Path) -> bool:
        """Download PDF and vector.zip for an HTML-format product."""
        folder_name = product["folder_name"]
        dest_dir    = activation_raw_dir / folder_name

        already_has_shp = dest_dir.exists() and any(dest_dir.rglob("*.shp"))
        already_has_pdf = dest_dir.exists() and any(dest_dir.rglob("*.pdf"))

        success = False

        # Download and extract vector ZIP
        if product["vector_url"] and not already_has_shp:
            zip_url  = product["vector_url"]
            zip_name = zip_url.split("/")[-1].split("?")[0] or f"{folder_name}_vector.zip"
            zip_path = TEMP_DIR / zip_name
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            print(f"        ↓  {zip_name}")
            if self._download_file(zip_url, zip_path):
                vector_dir = dest_dir / "VECTOR"
                vector_dir.mkdir(parents=True, exist_ok=True)
                if self._extract_zip(zip_path, vector_dir):
                    success = True
                zip_path.unlink(missing_ok=True)
        elif already_has_shp:
            success = True

        # Download PDF
        if product["pdf_url"] and not already_has_pdf:
            pdf_name = product["pdf_url"].split("/")[-1].split("?")[0]
            pdf_dest = dest_dir / pdf_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            print(f"        ↓  {pdf_name}")
            if self._download_file(product["pdf_url"], pdf_dest):
                success = True
        elif already_has_pdf:
            success = True

        return success

    # ── public interface ──────────────────────────────────────────────────

    def get_products(self, code: str) -> List[Dict]:
        """Return list of products for this activation (API or HTML method)."""
        if _emsr_num(code) >= API_THRESHOLD:
            return self._get_products_api(code)
        return self._get_products_html(code)

    def download_product(self, code: str, product: Dict,
                          activation_raw_dir: Path) -> bool:
        if _emsr_num(code) >= API_THRESHOLD:
            return self._download_api_product(code, product, activation_raw_dir)
        return self._download_html_product(code, product, activation_raw_dir)


# ─── PDF PARSER ──────────────────────────────────────────────────────────────

class PDFParser:
    """
    Extracts acquisition date and data source information from EMSR PDF maps.

    Handles three PDF eras:
      New   (EMSR656+):  clean text, "Data sources and analysis:" header,
                          "Pre-event image:" / "Post-event image:" labels.
      Middle (~200–655): spaced-font artefact ("Pre-even t im a ge:"),
                          recovered via get_text('blocks') which joins the box.
      Old   (<~200):     no pre/post labels; paragraph narrative text.
    """

    # Sensor name keywords used to identify old-format paragraph descriptions
    SENSOR_KEYWORDS = [
        "sentinel", "cosmo", "radarsat", "rapideye", "pleiades", "plé",
        "spot ", "worldview", "landsat", "terrasar", "sar", "copernicus",
        "digitalglobe", "airbus", "geoye", "kompsat",
    ]

    def find_pdf(self, product_dir: Path) -> Optional[Path]:
        """
        Recursively find the best PDF in a product folder.
        Prefers 'RTP' subfolder (ready-to-print) which contains data sources.
        Delineation products preferred over Reference.
        """
        # Prefer RTP subfolder (API format)
        for pdf in product_dir.rglob("RTP/*.pdf"):
            return pdf
        # Then any PDF with DELINEATION in name
        del_pdfs = [p for p in product_dir.rglob("*.pdf")
                    if "delineation" in p.name.lower() or "DEL" in p.name]
        if del_pdfs:
            return del_pdfs[0]
        # Fallback: any PDF
        all_pdfs = list(product_dir.rglob("*.pdf"))
        return all_pdfs[0] if all_pdfs else None

    def _get_blocks_text(self, pdf_path: Path) -> str:
        """
        Extract text using get_text('blocks') from all pages.
        Joins each block into a single line (fixes spaced-font artefact).
        Returns concatenated block texts separated by ' | '.
        """
        doc = fitz.open(str(pdf_path))
        parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            for blk in blocks:
                txt = blk[4].replace("\n", " ").strip()
                if txt:
                    parts.append(txt)
        doc.close()
        return " | ".join(parts)

    # ── date extraction ───────────────────────────────────────────────────

    # "Situation as of DD/MM/YYYY" in many languages, or standalone dates
    _DATE_PATTERNS = [
        r"[Ss]ituation\s+as\s+of\s+(\d{1,2}[/.]?\d{1,2}[/.]?\d{4})",
        r"[Ss]ituation\s+au\s+(\d{1,2}[/.]?\d{1,2}[/.]?\d{4})",
        r"[Ss]ituazione\s+al\s+(\d{1,2}[/.]?\d{1,2}[/.]?\d{4})",
        r"[Ss]tand\s+vom\s+(\d{1,2}[/.]?\d{1,2}[/.]?\d{4})",
        r"[Ff]lood\s*[-–]\s*[Ss]ituation\s+as\s+of\s+(\d{1,2}[/.]?\d{1,2}[/.]?\d{4})",
        r"[Ff]lood\s*[-–]\s*(\d{1,2}[/.]?\d{1,2}[/.]?\d{4})",
        r"as\s+of\s+(\d{1,2}[/.]?\d{1,2}[/.]?\d{4})",
    ]

    def extract_date(self, pdf_path: Path) -> Optional[str]:
        """
        Return YYYYMMDD string for the event date, or None.
        Tries all pages, all date patterns.
        """
        full_text = self._get_blocks_text(pdf_path)
        for pat in self._DATE_PATTERNS:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                raw = m.group(1)
                parsed = self._parse_date(raw)
                if parsed:
                    return parsed
        # Last resort: any date-like token
        for m in re.finditer(r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})", full_text):
            parsed = self._parse_date(m.group(0))
            if parsed:
                return parsed
        return None

    def _parse_date(self, raw: str) -> Optional[str]:
        raw = raw.replace(".", "/")
        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y%m%d")
            except ValueError:
                pass
        return None

    # ── data sources extraction ───────────────────────────────────────────

    def extract_sources(self, pdf_path: Path) -> Dict:
        """
        Extract pre/post-event image info from a PDF.

        Returns a dict with keys matching SOURCES_FIELDS.
        Missing values are empty strings.
        """
        result = {f: "" for f in SOURCES_FIELDS}
        result["pdf_file"] = pdf_path.name

        full_text = self._get_blocks_text(pdf_path)

        # ── locate the "data sources" block ──────────────────────────────
        src_block = self._find_sources_block(full_text)
        if not src_block:
            # Old format: try whole text
            src_block = full_text
        result["raw_sources_text"] = src_block[:1000]

        # ── try structured extraction (new + middle era) ──────────────────
        pre_text, post_texts = self._split_pre_post(src_block)

        # Middle-era PDFs: "Pre-event" section can appear BEFORE the "Data Sources"
        # map-legend header, so the narrowed src_block misses it.  Fall back to
        # searching the full document text.
        if not pre_text and not post_texts and src_block != full_text:
            pre_text, post_texts = self._split_pre_post(full_text)

        if pre_text:
            result.update(self._parse_image_entry(pre_text, prefix="pre_event_"))

        if post_texts:
            for i, pt in enumerate(post_texts[:2]):  # max 2 post-event sensors
                pfx = "post_event_" if i == 0 else "post_event_2"
                # adjust second prefix to match field names
                if i == 1:
                    d = self._parse_image_entry(pt, prefix="post_event_")
                    result["post_event_sensor2"]    = d.get("post_event_sensor", "")
                    result["post_event_date2"]      = d.get("post_event_date", "")
                    result["post_event_gsd2"]       = d.get("post_event_gsd_m", "")
                    result["post_event_cloud2"]     = d.get("post_event_cloud_pct", "")
                else:
                    result.update(self._parse_image_entry(pt, prefix="post_event_"))

        # ── old-format fallback: search whole paragraph ───────────────────
        if not result.get("post_event_sensor"):
            self._old_format_fallback(src_block, result)

        return result

    def _find_sources_block(self, text: str) -> Optional[str]:
        """
        Locate the data-sources section.
        Looks for 'Data source' header and takes text up to 'Base vector layers' / disclaimer.
        """
        m = re.search(
            r"[Dd]ata\s+source[s]?\s*(?:and\s+analysis)?\s*:?\s*(.+?)(?:Base\s+vector|Inset\s+map|Disclaimer|Population\s+data|Digital\s+Elevation|$)",
            text, re.DOTALL | re.IGNORECASE
        )
        return m.group(1).strip() if m else None

    def _split_pre_post(self, text: str) -> Tuple[Optional[str], List[str]]:
        """
        Split sources text into pre-event and post-event segments.
        Handles the spaced font: "Pre-even t im a ge:" → searchable via loose regex.
        """
        # Flexible regex for "Pre-event image:" allowing any spacing between chars
        pre_pat  = re.compile(
            r"[Pp]re[-\s]?even\s*t\s+im\s*a?\s*ge\s*:",
            re.IGNORECASE
        )
        post_pat = re.compile(
            r"[Pp]ost[-\s]?even\s*t\s+im\s*a?\s*ge\s*:",
            re.IGNORECASE
        )

        pre_m  = pre_pat.search(text)
        post_m = post_pat.search(text)

        if not pre_m and not post_m:
            return None, []

        pre_text = None
        if pre_m:
            end = post_m.start() if post_m else len(text)
            pre_text = text[pre_m.end():end].strip()

        post_texts = []
        if post_m:
            post_full = text[post_m.end():].strip()
            # Multiple post-event sensors are joined with " and " or " AND "
            # Split on "and Sentinel", "and RADARSAT", etc. (sensor-name junction)
            post_texts = self._split_multi_sensor(post_full)

        return pre_text, post_texts

    # Known satellite/sensor name prefixes used as split points in multi-sensor entries
    _SENSOR_PREFIXES = (
        "Sentinel", "RADARSAT", "COSMO", "Pl\u00e9iades", "Pleiades",
        "SPOT", "WorldView", "RapidEye", "Landsat", "TerraSAR",
        "GeoEye", "KOMPSAT", "ERS", "Alos", "ALOS", "Airbus",
    )

    def _split_multi_sensor(self, text: str) -> List[str]:
        """
        Split a post-event field that may have multiple sensors:
        "RADARSAT 2 ... (acquired on X) and Sentinel-1 ... (acquired on Y)"
        Only splits on " and " immediately followed by a known sensor-name prefix.
        This avoids false splits like "MacDonald, Dettwiler and Associates".
        Returns list of per-sensor text fragments.
        """
        prefix_pattern = "|".join(re.escape(p) for p in self._SENSOR_PREFIXES)
        parts = re.split(
            rf"\s+and\s+(?={prefix_pattern})",
            text, flags=re.IGNORECASE
        )
        return [p.strip() for p in parts if p.strip()]

    # ── sensor normalisation ──────────────────────────────────────────────

    # Ordered list of (pattern_on_collapsed_lowercase, canonical_name).
    # Applied after removing all spaces and lowercasing the raw sensor text.
    _SENSOR_MAP = [
        # COSMO-SkyMed (must come before generic "cosmo")
        (r"cosmo.*skymed.*sg|cosmo.*skym.*sg|csg[12]?",  "COSMO-SkyMed SG"),
        (r"cosmo.*skymed|cosmo.*skym|cosmosky",          "COSMO-SkyMed"),
        # RADARSAT
        (r"radarsatconstellation|radarsatcm",             "RADARSAT Constellation"),
        (r"radarsat.*2|radarsat-2|radars.*at2",           "RADARSAT-2"),
        (r"radarsat",                                     "RADARSAT-2"),
        # Sentinel-1 variants (sentin[ae]l handles "Sentinal" typo in PDFs)
        (r"sentin[ae]l.*1.*a/?b|sentin[ae]l.*1[ab]/[ab]", "Sentinel-1A/B"),
        (r"sentin[ae]l.*1.*a",                             "Sentinel-1A"),
        (r"sentin[ae]l.*1.*b",                             "Sentinel-1B"),
        (r"sentin[ae]l.*1",                                "Sentinel-1"),
        # Sentinel-2 variants
        (r"sentin[ae]l.*2.*a/?b|sentin[ae]l.*2[ab]/[ab]", "Sentinel-2A/B"),
        (r"sentin[ae]l.*2.*a",                             "Sentinel-2A"),
        (r"sentin[ae]l.*2.*b",                             "Sentinel-2B"),
        (r"sentin[ae]l.*2",                                "Sentinel-2"),
        # Generic Sentinel (no number) — should be rare
        (r"sentin[ae]l",                                   "Sentinel"),
        # Pléiades Neo (must come before plain Pléiades)
        # pl.{1,4}des covers pléiades/pleiades/pleiàdes/plèiades accent variants
        (r"pl.{1,4}des.*neo",                             "Pléiades Neo"),
        # Pléiades-1A/B variants
        (r"pl.{1,4}des.*1.*a/?b|pl.{1,4}des.*1[ab]/[ab]", "Pléiades-1A/B"),
        (r"pl.{1,4}des.*1.*a",                            "Pléiades-1A"),
        (r"pl.{1,4}des.*1.*b",                            "Pléiades-1B"),
        (r"pl.{1,4}des",                                  "Pléiades"),
        # SPOT
        (r"spot.*6/?7|spot6/7",                           "SPOT-6/7"),
        (r"spot.*6",                                      "SPOT-6"),
        (r"spot.*7",                                      "SPOT-7"),
        (r"spot",                                         "SPOT"),
        # WorldView (worl.?view handles the common "WorlView" typo in PDFs)
        (r"worl.?view.*3",                                "WorldView-3"),
        (r"worl.?view.*2",                                "WorldView-2"),
        (r"worl.?view.*1",                                "WorldView-1"),
        (r"worl.?view",                                   "WorldView"),
        # Landsat
        (r"landsat.*9",                                   "Landsat-9"),
        (r"landsat.*8",                                   "Landsat-8"),
        (r"landsat",                                      "Landsat"),
        # TerraSAR-X
        (r"terrasar|terrasarx",                           "TerraSAR-X"),
        # GeoEye
        (r"geoeye.*1|geoeye-1",                           "GeoEye-1"),
        (r"geoeye",                                       "GeoEye-1"),
        # RapidEye
        (r"rapideye",                                     "RapidEye"),
        # ESRI / background imagery (keep concise)
        # "imagery" alone (fragment from old-format fallback) also maps here
        (r"esri.*world.*imag|esriworldimage|^imag",       "ESRI World Imagery"),
        # ICEYE (SAR satellites; IDs are IE00, IE01, IE02 …)
        (r"iceye|^ie\d{2}",                              "ICEYE"),
        # PlanetScope / SkySat
        (r"planetscope|planet.*scope",                    "PlanetScope"),
        (r"skysat|sky.*sat",                              "SkySat"),
        # Deimos
        (r"deimos.*2|deimos-2",                           "Deimos-2"),
        (r"deimos",                                       "Deimos-2"),
        # Others
        (r"saocom",                                       "SAOCOM"),
        (r"alos.*2|alos-2",                               "ALOS-2"),
        (r"alos",                                         "ALOS-2"),
        (r"paz",                                          "PAZ"),
        (r"kompsat",                                      "KOMPSAT"),
        (r"digitalglobe",                                 "DigitalGlobe"),
        (r"mapgenie",                                     "MapGenie"),
        (r"aerial",                                       "Aerial"),
        (r"uas|uav",                                      "UAS/UAV"),
    ]

    # Compiled once at class level
    _SENSOR_MAP_RE = [(re.compile(p), name) for p, name in _SENSOR_MAP]

    def _normalize_sensor(self, raw: str) -> str:
        """
        Clean a raw sensor string down to just the canonical satellite name.

        Steps:
          1. Strip leading label garbage ("Post-event image:", "ost-event ", ":")
          2. Cut at copyright / attribution markers (©, @, "Data and", "provided", etc.)
          3. Collapse all whitespace (fixes spaced-font: "R ADAR S AT" → "RADARSAT")
          4. Match against _SENSOR_MAP (longest/first wins) → canonical name
          5. If no map match, return lightly-cleaned original
        """
        s = raw.strip()

        # 1. Strip leading label garbage
        s = re.sub(
            r"^(?:pre-?event\s+(?:image\s*:?\s*)?|post-?event\s+(?:image\s*:?\s*)?|"
            r"ost-?event\s+|ost\s+event\s+|image\s*:\s*|:\s*)",
            "", s, flags=re.IGNORECASE
        ).strip()

        # 2. Cut at attribution / trailing noise markers
        for marker in (
            r"\s*©", r"\s*@\s*(?!Microsoft)", r"\bdata\s+and\s+prod",
            r"\bprovided\s+under", r"\bcourtesy\s+of", r"\bright[s]?\s+reserved",
            r"\bASI\b", r"\be-GEOS\b", r"\bMacDonald\b", r"\bASTRIUM\b",
            r"\bInfoterra\b", r"\bCNES\b(?!\s+\d)",
            r"\s*\(a\s*c\s*q\s*u",   # (acquired on …) — spaced-font variant included
            r"\s+satellite\s+imag",   # "PAZ satellite image" → "PAZ"
            r"\s+radar\s+imag",       # "COSMO-SkyMed radar image" → "COSMO-SkyMed"
            r"\s+post-?event\b",      # trailing "post-event" noise
        ):
            s = re.split(marker, s, maxsplit=1, flags=re.IGNORECASE)[0].strip()

        # Strip trailing bare year (e.g. "Sentinel-2A/B 2018") or "(2018)"
        s = re.sub(r"\s*\(?\d{4}\)?\s*$", "", s).strip()
        s = s.rstrip(" ,-.")

        # 3. Collapse all spaces for pattern matching (keeps original for fallback)
        collapsed = re.sub(r"\s+", "", s).lower()

        # 4. Match against sensor map
        for pattern, canonical in self._SENSOR_MAP_RE:
            if pattern.search(collapsed):
                return canonical

        # 5. No match — return first 1-3 words only (Solution 2: keyword extraction)
        cleaned = re.sub(r"\s+", " ", s).strip()
        if len(cleaned) >= 4:
            # Take only first 1-3 words to avoid long sentences
            words = cleaned.split()
            if len(words) <= 3:
                return cleaned
            # For longer strings, take first 3 words or first word if it looks like a sensor
            first_word = words[0]
            # If first word is long enough and looks like a sensor name, use it
            if len(first_word) >= 5 and not first_word.lower() in ['image', 'imagery', 'data', 'from']:
                # Take up to 3 words for compound names like "RADARSAT Constellation Mission"
                return " ".join(words[:min(3, len(words))])
            return cleaned[:30].strip()  # Limit to 30 chars max

        # Fragment too short (e.g. "ry" from "World Imag ery © ESRI …").
        # Last resort: try the sensor map on the full original string.
        full_collapsed = re.sub(r"\s+", "", raw.lower())
        for pattern, canonical in self._SENSOR_MAP_RE:
            if pattern.search(full_collapsed):
                return canonical
        return ""

    def _parse_image_entry(self, text: str, prefix: str) -> Dict:
        """
        Parse a single image-entry text block for:
          - sensor name
          - acquisition date  (DD/MM/YYYY or YYYY-MM-DD)
          - GSD in metres
          - cloud coverage %
        """
        result: Dict[str, str] = {}
        key_sensor = prefix + "sensor"
        key_date   = prefix + "date"
        key_gsd    = prefix + "gsd_m"
        key_cloud  = prefix + "cloud_pct"

        # Sensor name: text before copyright "©", year "(20XX)", or "(acquired"
        # e.g. "RADARSAT 2 Data and products © MacDonald..." → "RADARSAT 2"
        # e.g. "Sentinel-2A/B (2022) (acquired..." → "Sentinel-2A/B"
        sensor_m = re.match(
            r"^(.+?)(?:\s*©|\s*\(\d{4}\)|\s*\(acquired|\s+acquired|$)",
            text, re.IGNORECASE
        )
        if sensor_m:
            raw_sensor = sensor_m.group(1).strip().rstrip("©,- ")
            result[key_sensor] = self._normalize_sensor(raw_sensor)

        # Acquisition date — "acquired on DD/MM/YYYY"
        # Allow optional spaces inside "acquired" (spaced-font PDF artefact: "a cquired")
        date_m = re.search(
            r"a\s*c\s*q\s*u\s*i\s*r\s*e\s*d\s+on\s+(\d{1,2}[/.]?\d{1,2}[/.]?\d{4})",
            text, re.IGNORECASE
        )
        if not date_m:
            # Fallback: any standalone DD/MM/YYYY in the block
            date_m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", text)
        if date_m:
            parsed = self._parse_date(date_m.group(1))
            result[key_date] = parsed or date_m.group(1)

        # GSD / resolution
        gsd_m = re.search(
            r"(?:GSD|resolution)\s*\(?(\d+(?:[.,]\d+)?)\s*m",
            text, re.IGNORECASE
        )
        if gsd_m:
            result[key_gsd] = gsd_m.group(1).replace(",", ".")

        # Cloud coverage
        cloud_m = re.search(
            r"(?:approx\.\s*)?(\d+(?:[.,]\d+)?)\s*%\s*cloud",
            text, re.IGNORECASE
        )
        if cloud_m:
            result[key_cloud] = cloud_m.group(1).replace(",", ".")

        return result

    # SAR sensors are almost always the post-event image in EMSR products
    _SAR_SENSORS = ["cosmo", "radarsat", "sentinel-1", "terrasar", "ers-", "alos"]
    # Optical sensors used as pre-event/background
    _OPT_SENSORS = ["rapideye", "worldview", "pleiades", "spot ", "landsat",
                    "sentinel-2", "digitalglobe", "esri", "geoeye", "kompsat"]

    def _old_format_fallback(self, text: str, result: Dict):
        """
        For old-format PDFs (no Pre-event/Post-event labels):
        heuristic approach — SAR sensors → post-event, optical → pre-event.

        Also handles text like:
          "visible from a COSMO-SkyMed image (31/08/2012)"
          "pre-event satellite images RapidEye © RapidEye (acquired on ...)"
        """
        sentences = re.split(r"[.!?\n]|\s*\|\s*", text)

        for sent in sentences:
            s_low = sent.lower()
            if not any(kw in s_low for kw in self.SENSOR_KEYWORDS):
                continue

            # Extract date (parenthesised or after "acquired on")
            date_str = ""
            date_m = re.search(
                r"(?:acquired\s+on\s+)?(\d{1,2}/\d{1,2}/\d{4})", sent
            )
            if date_m:
                parsed = self._parse_date(date_m.group(1))
                date_str = parsed or date_m.group(1)

            # Extract GSD
            gsd_str = ""
            gsd_m = re.search(r"(\d+(?:\.\d+)?)\s*m\s*(?:resolution|GSD)", sent, re.I)
            if gsd_m:
                gsd_str = gsd_m.group(1)

            # Extract sensor name: text near the first sensor keyword
            sensor = ""
            for kw in self.SENSOR_KEYWORDS:
                pos = s_low.find(kw)
                if pos >= 0:
                    # Grab a window around the keyword
                    start = max(0, pos - 10)
                    chunk = sent[start:pos + 60].strip()
                    # Clean up: stop at "(" or date
                    m = re.match(r"^([^(\d]+)", chunk)
                    if m:
                        sensor = self._normalize_sensor(m.group(1).strip())
                    break

            # Classify as post-event (SAR) or pre-event (optical/background)
            is_sar = any(kw in s_low for kw in self._SAR_SENSORS)
            is_opt = any(kw in s_low for kw in self._OPT_SENSORS)
            explicit_pre  = any(kw in s_low for kw in ["pre-event", "pre event", "background"])
            explicit_post = any(kw in s_low for kw in ["post-event", "post event", "visible from"])

            if (is_sar or explicit_post) and not result.get("post_event_sensor"):
                result["post_event_sensor"] = sensor
                result["post_event_date"]   = date_str
                if gsd_str:
                    result["post_event_gsd_m"] = gsd_str
            elif (is_opt or explicit_pre) and not result.get("pre_event_sensor"):
                result["pre_event_sensor"] = sensor
                result["pre_event_date"]   = date_str
                if gsd_str:
                    result["pre_event_gsd_m"] = gsd_str


# ─── SHAPEFILE FINDER ────────────────────────────────────────────────────────

class ShapefileFinder:
    """
    Locates aoi and event (flood extent) shapefiles inside a product folder.
    Handles both old (VECTOR/ subfolder) and new (flat) layouts.
    """

    @staticmethod
    def _norm(name: str) -> str:
        return name.lower().replace("_", "").replace("-", "").replace(" ", "")

    @staticmethod
    def _search_dir(product_dir: Path) -> Path:
        """Returns the directory to search: VECTOR/ subfolder if it exists, else product_dir."""
        # API format: shapefiles are in a versioned subfolder (EMSR657_AOI01_..._v1/)
        # HTML format: VECTOR/ subfolder
        vector = product_dir / "VECTOR"
        if vector.is_dir():
            return vector
        # For API format, look one level deeper for versioned folders
        for sub in product_dir.iterdir():
            if sub.is_dir() and any(sub.glob("*.shp")):
                return sub
        return product_dir

    def _find_shp(self, directory: Path,
                  keywords: List[str],
                  exclude: List[str] = None,
                  require_any: List[str] = None) -> Optional[Path]:
        exclude    = exclude or []
        require_any = require_any or []
        directory_list = [directory]
        # Also search one level deeper (versioned folders like EMSR657_AOI01_..._v1/)
        directory_list += [d for d in directory.iterdir() if d.is_dir()]

        for search_dir in directory_list:
            if not search_dir.is_dir():
                continue
            for f in search_dir.iterdir():
                if f.suffix.lower() != ".shp":
                    continue
                norm = self._norm(f.stem)
                if not all(self._norm(kw) in norm for kw in keywords):
                    continue
                if any(ex.lower() in f.stem.lower() for ex in exclude):
                    continue
                if require_any and not any(r.lower() in f.stem.lower() for r in require_any):
                    continue
                return f
        return None

    def find_aoi(self, product_dir: Path) -> Optional[Path]:
        search = self._search_dir(product_dir)
        # Primary: modern naming (AreaOfInterest, area_of_interest, areaOfInterestA)
        f = self._find_shp(search, ["area", "interest"], exclude=["_line", "_point"])
        if f:
            return f
        # Fallback: old format names AOI as _AOI1_DTL2.shp etc.
        return self._find_shp(search, ["aoi"], exclude=["_line", "_point"])

    def find_event(self, product_dir: Path) -> Optional[Path]:
        search = self._search_dir(product_dir)
        for kws in [
            (["observed", "event"], ["_line", "_point"]),
            (["crisis", "event"],   ["_line", "_point"]),
            (["crisis", "information"], ["_line", "_point"]),
        ]:
            f = self._find_shp(search, kws[0], exclude=kws[1])
            if f:
                return f
        return None


# ─── ACTIVATION REORGANIZER ───────────────────────────────────────────────────────────

class ActivationReorganizer:
    """
    Copies shapefiles from a raw product folder into standardized format:
      data/activations/{act_folder_name}/
        aoi/aoi.{shp,shx,dbf,prj,...}            mapped area-of-interest footprint
        flood_extent/event.{shp,...}             observed flood delineation (label)

    Only these two CEMS components are kept. The pre-event hydrography (hydroA)
    is not used: CEMS stopped shipping it after early 2023, so the permanent
    water layer is defined from ESA WorldCover alone (Step 4) for every event.
    """

    def __init__(self):
        self.finder = ShapefileFinder()

    def convert(self, product_dir: Path, act_folder: Path) -> Tuple[bool, Dict[str, bool], str]:
        """Returns (overall_success, {has_aoi, has_event}, message)."""
        if act_folder.exists() and (act_folder / "aoi" / "aoi.shp").exists():
            return True, {"aoi": True, "event": True}, "already done"

        aoi_shp   = self.finder.find_aoi(product_dir)
        event_shp = self.finder.find_event(product_dir)

        if not aoi_shp:
            return False, {"aoi": False, "event": bool(event_shp)}, "missing aoi shapefile"

        act_folder.mkdir(parents=True, exist_ok=True)

        self._copy_shp(aoi_shp,   act_folder / "aoi",          "aoi")
        self._copy_shp(event_shp, act_folder / "flood_extent", "event")

        return True, {"aoi": True, "event": True}, "ok"

    @staticmethod
    def _copy_shp(src: Path, dest_dir: Path, new_stem: str):
        dest_dir.mkdir(parents=True, exist_ok=True)
        for ext in SHAPEFILE_EXTENSIONS:
            candidate = src.parent / f"{src.stem}{ext}"
            if candidate.exists():
                shutil.copy2(candidate, dest_dir / f"{new_stem}{ext}")


# ─── UTILITIES ───────────────────────────────────────────────────────────────

def write_activations_catalog(status: 'StatusTracker', csv_path: Path):
    """Write activations.csv — catalog: folder name, date, pre/post sensors."""
    rows = []
    for (emsr_code, product_folder), rec in status._data.items():
        if rec.get("reorg_converted") != "yes":
            continue
        act_folder_rel = rec.get("act_folder", "")
        folder_name = act_folder_rel.split("/", 1)[-1] if "/" in act_folder_rel else act_folder_rel
        if not folder_name:
            continue
        rows.append({
            "folder_name":        folder_name,
            "emsr_code":          emsr_code,
            "event_date":         rec.get("event_date", ""),
            "pre_event_sensor":   rec.get("pre_event_sensor", ""),
            "post_event_sensors": rec.get("post_event_sensors", ""),
        })
    rows.sort(key=lambda r: r["folder_name"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["folder_name", "emsr_code", "event_date",
                                                "pre_event_sensor", "post_event_sensors"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Activations catalog → {csv_path}  ({len(rows)} records)")


def fetch_flood_codes_by_date(date_start: str, date_end: str) -> List[Dict]:
    """
    Fetch all flood activations from Copernicus between date_start and date_end.

    Uses the mapping.emergency.copernicus.eu API which covers the full archive
    (EMSR001 onward). The API does not support server-side date/category filtering,
    so we page through all results and filter client-side.

    Returns list of dicts sorted by activationTime:
      {"code": "EMSR845", "name": "Flood in Italy", "date": "2023-01-15",
       "countries": ["Italy"], "n_products": 10}
    """
    API_URL  = "https://mapping.emergency.copernicus.eu/activations/api/activations/"
    session  = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    results  = []
    url      = f"{API_URL}?ordering=activationTime&limit=100"

    print("  Fetching activation list from Copernicus ...", end=" ", flush=True)
    page = 0
    while url:
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"\n  ! API error: {e}")
            break

        for act in data.get("results", []):
            category = act.get("category", {})
            cat_name = category.get("name", "") if isinstance(category, dict) else str(category)
            if "flood" not in cat_name.lower():
                continue

            act_date = (act.get("activationTime") or "")[:10]
            if not act_date:
                continue
            if act_date < date_start or act_date > date_end:
                continue

            countries = [c.get("short_name", c) if isinstance(c, dict) else c
                         for c in act.get("countries", [])]
            results.append({
                "code":       act["code"],
                "name":       act.get("name", ""),
                "date":       act_date,
                "countries":  countries,
                "n_products": act.get("n_products", 0),
            })

        page += 1
        url = data.get("next")

    print(f"done ({page} pages)")
    return results


def _emsr_num(code: str) -> Optional[int]:
    m = re.search(r"EMSR(\d+)", code)
    return int(m.group(1)) if m else None


def _build_act_name(emsr_code: str, product_folder: str, date: Optional[str]) -> str:
    """
    Build canonical standardized folder name:
      EMSR657_AOI01_DaugavaRiver_DEL_MONIT01_20230402
    or without date if unknown:
      EMSR657_AOI01_DaugavaRiver_DEL_MONIT01_NODATE
    """
    suffix = date if date else "NODATE"
    return f"{emsr_code}_{product_folder}_{suffix}"


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt and download immediately")
    args = parser.parse_args()

    tee = _setup_logging()

    print("=" * 72)
    print("  EMSR Flood Download + Activation Reorganizer  (Script 1)")
    print(f"  BASE_DIR   : {BASE_DIR}")
    print(f"  Date range : {DATE_START}  →  {DATE_END}")
    print("=" * 72)
    print()

    # Create directories
    for d in [RAW_DIR, ACT_DIR, META_DIR, TEMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: fetch flood activations in date range ─────────────────────────
    flood_acts = fetch_flood_codes_by_date(DATE_START, DATE_END)

    if not flood_acts:
        print(f"\n  No flood activations found between {DATE_START} and {DATE_END}.")
        return

    print(f"\n  Found {len(flood_acts)} flood activations:\n")
    print(f"  {'Code':<12} {'Date':<12} {'Countries':<28} {'Products':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*28} {'-'*8}")
    for act in flood_acts:
        countries = ", ".join(act["countries"])[:27]
        print(f"  {act['code']:<12} {act['date']:<12} {countries:<28} {act['n_products']:>8}")
    total_products = sum(a["n_products"] for a in flood_acts)
    print(f"\n  Total: {len(flood_acts)} activations, ~{total_products} products\n")

    if not args.yes:
        proceed = input("  Proceed with download? [y/N] ").strip().lower()
        if proceed != "y":
            print("  Aborted.")
            return

    # Initialise helpers
    META_DIR.mkdir(parents=True, exist_ok=True)
    config.migrate_csv_names()  # rename any old-named metadata files in place
    status   = StatusTracker(STATUS_CSV)
    dl       = EMSRDownloader()
    pdf_p    = PDFParser()
    reorg_conv = ActivationReorganizer()

    # Stats
    stats = dict(downloaded=0, reorg_ok=0, reorg_fail=0, errors=0)

    total = len(flood_acts)

    for idx, act in enumerate(flood_acts, 1):
        code       = act["code"]
        event_type = act["name"]
        print(f"\n[{idx}/{total}] {code}  —  {event_type}")

        # ── Step 2: get product list ──────────────────────────────────────
        products = dl.get_products(code)
        if not products:
            print(f"  !  no products found for {code}")
            time.sleep(REQUEST_DELAY)
            continue

        print(f"  → {len(products)} products")
        activation_raw_dir = RAW_DIR / code

        for prod in products:
            folder = prod["folder_name"]
            print(f"\n    [{folder}]")

            # ── Step 3: download raw ──────────────────────────────────────
            if status.is_raw_done(code, folder):
                print(f"      ↩  raw already downloaded")
            else:
                ok = dl.download_product(code, prod, activation_raw_dir)
                if not ok:
                    print(f"      ✗  download failed")
                    status.upsert({"emsr_code": code, "product_folder": folder,
                                   "event_type": event_type,                                    "raw_downloaded": "no",
                                   "skipped_reason": "download failed"})
                    stats["errors"] += 1
                    time.sleep(REQUEST_DELAY)
                    continue
                stats["downloaded"] += 1
                status.upsert({"emsr_code": code, "product_folder": folder,
                               "event_type": event_type,                                "raw_downloaded": "yes"})

            product_dir = activation_raw_dir / folder

            # ── Step 4: find PDF (mandatory) + extract date ───────────────
            pdf_path = pdf_p.find_pdf(product_dir)

            if not pdf_path:
                print(f"      –  skipping: no PDF found (not a map product)")
                status.upsert({"emsr_code": code, "product_folder": folder,
                               "event_type": event_type,                                "raw_downloaded": "yes",
                               "pdf_found": "no",
                               "skipped_reason": "no pdf"})
                time.sleep(REQUEST_DELAY)
                continue

            print(f"      PDF: {pdf_path.name}")

            # ── Step 4b: skip reference/overview products (no event shp) ──
            if not reorg_conv.finder.find_event(product_dir):
                print(f"      –  skipping: no event shapefile (reference/overview product)")
                status.upsert({"emsr_code": code, "product_folder": folder,
                               "event_type": event_type,                                "raw_downloaded": "yes",
                               "pdf_found": "yes",
                               "skipped_reason": "no event shapefile"})
                time.sleep(REQUEST_DELAY)
                continue

            date_str = pdf_p.extract_date(pdf_path)
            if date_str:
                print(f"      Date: {date_str}")
            else:
                date_str = prod.get("delivery_date")
                if date_str:
                    print(f"      Date (delivery fallback): {date_str}")
                else:
                    print(f"      ! no date found")

            status.upsert({"emsr_code": code, "product_folder": folder,
                           "pdf_found":       "yes",
                           "date_extracted":  "yes" if date_str else "no",
                           "event_date":      date_str or ""})

            # ── Step 5: extract data sources from PDF ─────────────────────
            existing = status.get(code, folder)
            if not existing or not existing.get("pre_event_sensor") and not existing.get("post_event_sensors"):
                src = pdf_p.extract_sources(pdf_path)
                post_parts = [src.get("post_event_sensor", ""), src.get("post_event_sensor2", "")]
                pre  = src.get("pre_event_sensor", "").strip()
                post = ", ".join(p.strip() for p in post_parts if p.strip())
                print(f"      Sources: pre={pre!r}  post={post!r}")
                status.upsert({"emsr_code": code, "product_folder": folder,
                               "pre_event_sensor": pre, "post_event_sensors": post})
            else:
                print(f"      ↩  sources already extracted")

            # ── Step 6: reorganization ─────────────────────────────────────
            if status.is_reorganized(code, folder):
                print(f"      ↩  already reorganized")
                stats["reorg_ok"] += 1
                continue

            act_name   = _build_act_name(code, folder, date_str)
            act_folder = ACT_DIR / code / act_name   # EMSR parent level added

            ok, flags, msg = reorg_conv.convert(product_dir, act_folder)

            if ok:
                print(f"      ✓  reorganized → {code}/{act_name}  "
                      f"[aoi={flags['aoi']} flood_extent={flags['event']}]")
                stats["reorg_ok"] += 1
                status.upsert({
                    "emsr_code":      code,
                    "product_folder": folder,
                    "act_folder":     f"{code}/{act_name}",
                    "reorg_converted":  "yes",
                    "has_aoi":        "yes" if flags["aoi"]   else "no",
                    "has_event":      "yes" if flags["event"] else "no",
                })
            else:
                print(f"      ✗  reorganization failed: {msg}")
                stats["reorg_fail"] += 1
                status.upsert({
                    "emsr_code":      code,
                    "product_folder": folder,
                    "reorg_converted":  "no",
                    "skipped_reason": msg,
                    "has_aoi":        "yes" if flags.get("aoi")   else "no",
                    "has_event":      "yes" if flags.get("event") else "no",
                })

            time.sleep(REQUEST_DELAY)

    # ── Cleanup temp dir ──────────────────────────────────────────────────────
    if TEMP_DIR.exists():
        for f in TEMP_DIR.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
        try:
            TEMP_DIR.rmdir()
        except OSError:
            pass

    # ── Write first-draft catalog ─────────────────────────────────────────────
    write_activations_catalog(status, CATALOG_CSV)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Flood activations   : {total}")
    print(f"  Products downloaded : {stats['downloaded']}")
    print(f"  reorganizations ok  : {stats['reorg_ok']}")
    print(f"  reorganizations fail: {stats['reorg_fail']}")
    print(f"  Errors              : {stats['errors']}")
    print()
    print(f"  Activations catalog → {CATALOG_CSV}")
    print(f"  standardized folders         → {ACT_DIR}")
    print("=" * 72)

    tee.close()


if __name__ == "__main__":
    main()
