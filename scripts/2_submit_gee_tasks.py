#!/usr/bin/env python3
"""
Script 2: Submit GEE Export Tasks for Flood Activations

For each activation in data/activations/ (output of Script 1), this
script submits Google Earth Engine export tasks to Google Drive.

7 multi-band GeoTIFFs per activation (7 GEE tasks):
  S1_VV_VH.tif        2 bands   Sentinel-1 VV/VH median composite (30-90 days pre-event)
  land_cover.tif      2 bands   NDVI + NDBI from S2 with temporal fallback (both at 10m)
  MERIT.tif           4 bands   MERIT Hydro (elevation, flow dir, UDA, HAND)
  Soil.tif            2 bands   SoilGrids topsoil clay + sand
  ESA_WorldCover_PermanentWater.tif  1 band  ESA WorldCover 2021 permanent water
  Precipitation_{first}_{last}.tif  30 bands  GPM-IMERG V07 daily precip (30 antecedent days, event excluded)
  SoilMoisture_{first}_{last}.tif   30 bands  SMAP L4 daily soil moisture (30 antecedent days, NASA/SMAP/SPL4SMGP/008)

Temporal Fallback Strategy (Smart Iterative with Real Coverage Checking):

  S1 (SAR):
    Iterative approach with REAL coverage measurement:
    - Try 15 days → calculate actual coverage with .getInfo() → print %
    - If coverage < 99%: Try 30 days → fill gaps → measure again
    - If coverage < 99%: Try 45 days → fill gaps → measure again
    - If coverage < 99%: Try 60 days → fill gaps → measure again
    - STOPS EARLY when coverage sufficient (saves computation!)

    Why This is SMARTER:
      ✓ Uses .getInfo() to get ACTUAL coverage percentage (not blind server-side)
      ✓ Stops early when sufficient (doesn't waste computation on unneeded windows)
      ✓ Prints real coverage values (user visibility into what's happening)
      ✓ Learned from proven production code (download_gee_layers.py)

  S2_indices (NDVI/NDBI):
    Iterative approach with REAL coverage measurement:

    Step 1 - Current Year (S2_WINDOW_DAYS before event, default 30 days):
      For each cloud threshold (30% → 40% → 50% → 60% → 70%):
        - Build composite
        - Calculate ACTUAL coverage with .getInfo()
        - Print real coverage percentage
        - If coverage >= 98%: STOP (coverage sufficient!)
        - If coverage < 98%: Try next cloud threshold

    Step 2 - Seasonal Fallback (only if all current year thresholds insufficient):
      - Check if coverage < 98% after trying all current year thresholds
      - If yes: Use same iterative approach with seasonal period
        Year Selection Strategy:
          - 2017-2024: Use NEXT year (more data available ahead)
          - 2025+: Use PREVIOUS year (recent events)
      - Fill current year gaps with seasonal data

    Why This is SMARTER:
      ✓ Uses .getInfo() to get ACTUAL coverage percentage (not blind server-side)
      ✓ Stops early when sufficient (doesn't build unneeded composites)
      ✓ Prints real coverage values for each step (user visibility)
      ✓ Only builds seasonal if truly needed
      ✓ Learned from proven production code (download_gee_layers.py)

    Problem Solved:
      - Large AOIs covered by multiple S2 tiles
      - Can't assume full coverage even with many high-cloud images
      - Real coverage check (not hypothetical) determines what to do next
      - Avoids unnecessary computation (saves time & cost)
      - User can see exactly what's happening (not a black box)

    Output: 2-band stacked TIF (NDVI, NDBI) with NO date information in bands

Band names inside each TIF encode the layer name / date, e.g.:
  Precipitation.tif → bands named 'Precipitation_20230323' ... 'Precipitation_20230401'
  MERIT.tif         → bands named 'Elevation', 'FlowDirection', 'UDA', 'HAND'

Download tracking:
  data/metadata/2_gee_export_status.csv tracks layer availability per activation:
    - "NA" = No GEE images available (won't retry)
    - "no" = Not yet submitted / not available in exports
    - "yes" = File exists and passes validation
  Activations with any "NA" layer are skipped automatically.
  Activations without a flood-extent polygon are skipped.

Modes:
  Edit SUBMIT_TO_GEE in CONFIG section to "yes" or "no"
  python 2_submit_gee_tasks.py                  SUBMIT: based on SUBMIT_TO_GEE config
  python 2_submit_gee_tasks.py --update-tracking  UPDATE: only update tracking CSV

After GEE completes (typically hours):
  1. Run Script 3 (3_download_gee_exports.py) to download from Google Drive
  2. Downloaded files will be in: data/GEE_exports/{act_folder_name}/{layer}.tif
  3. Run Script 4 to validate exports and produce flood_dataset.csv
"""

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SUBMIT_TO_GEE    = "yes"             # yes or no - whether to submit tasks to GEE
# Temporal-layer length (precip / soil moisture days) and which layers to export
# are now set in config.py (N_DAYS_OVERRIDE, LAYER_TOGGLES).

# TEST MODE: Process only first activation folder (for testing)
TEST_MODE = False                   # Set to True to process only first valid activation

# Sentinel-1: progressive temporal windows (days before event)
# Will try 15d, then 30d, then 45d, then 60d to achieve target coverage
S1_WINDOWS = [15, 30, 45, 60]       # Progressive windows
S1_TARGET_COVERAGE = 0.99           # Target 99% coverage

# Sentinel-2: Simple approach with smart seasonal fallback
S2_WINDOW_DAYS = 30                 # Temporal window (days before event, configurable)
S2_CLOUD_INITIAL = 30               # Initial cloud threshold (user configurable: 10, 20, 30, etc.)
S2_CLOUD_PROGRESSIVE = [40, 50, 60, 70] # Progressive relaxation within the window
S2_COVERAGE_THRESHOLD = 0.98        # If coverage < 98%, add seasonal fallback

# Seasonal fallback (only triggered if current year has >1% gaps)
S2_USE_SEASONAL_FALLBACK = True     # Enable/disable seasonal fallback
S2_SEASONAL_STRATEGY = 'auto'       # 'auto' = next year for early events, prev year for recent

# GEE export pixel size (degrees) - fine reference grid for the 10 m layers.
# S1, S2 indices and ESA_PW are exported on this grid. MERIT (90 m) and SoilGrids
# (250 m) are exported on their own native-resolution grid over the same AOI.
PIXEL_DEG        = 0.0001           # ~10m at equator (S1, S2, ESA_PW)

# Temporal layers (precip / soil moisture) are exported at their NATIVE coarse
# resolution over an AOI bbox expanded by config.TEMPORAL_BUFFER_DEG, so every
# patch later has a full spatial neighbourhood grid around it.
PRECIP_DEG        = 0.1             # GPM-IMERG V07 native resolution (~11.1 km)
SMAP_DEG          = 0.09516         # SMAP SPL4SMGP native resolution (~9.5 km)

REQUEST_DELAY    = 5                # seconds between GEE task submissions
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import math
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config
from config import enabled_layers, enabled_keys

try:
    import ee
except ImportError:
    print("ERROR: earthengine-api not found.")
    print("Install it with: pip install earthengine-api")
    print("Then authenticate with: earthengine authenticate")
    sys.exit(1)

try:
    import geopandas as gpd
except ImportError:
    print("ERROR: geopandas not found in this environment.")
    sys.exit(1)

try:
    import rasterio
    import rasterio.mask
    import numpy as np
except ImportError:
    print("ERROR: rasterio not found in this environment.")
    sys.exit(1)


# ─── PATH SETUP ──────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = BASE_DIR / "data"
META_DIR     = DATA_DIR / "metadata"
ACTIVATIONS_DIR = DATA_DIR / "activations" / "activations_reorganized"
GEE_EXPORTS_DIR = DATA_DIR / "GEE_exports"
GEE_TASKS_CSV            = config.CSV_GEE_EXPORT_STATUS

# Layers this run exports, in registry order, after applying config toggles.
ALL_LAYERS = enabled_keys()


# ─── DOWNLOAD TRACKER ────────────────────────────────────────────────────────

class DownloadTracker:
    """Reads and writes download_tracking.csv."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._data: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if not self.csv_path.exists():
            return
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._data[row["folder_name"]] = row

    def get(self, folder_name: str) -> Optional[Dict]:
        return self._data.get(folder_name)

    def has_na(self, folder_name: str) -> bool:
        """Check if activation has any NA layers."""
        rec = self.get(folder_name)
        if not rec:
            return False
        for layer in ALL_LAYERS:
            if rec.get(layer) == "NA":
                return True
        return False

    def needs_submission(self, folder_name: str, layer: str) -> bool:
        """Check if a layer needs submission (status is 'no')."""
        rec = self.get(folder_name)
        if not rec:
            return True  # No record = needs submission
        status = rec.get(layer, "no")
        return status == "no"  # Only submit if status is "no"

    def upsert(self, record: Dict):
        """Update or insert a record."""
        folder_name = record["folder_name"]
        if folder_name in self._data:
            self._data[folder_name].update(record)
        else:
            self._data[folder_name] = record
        self._flush()

    def mark_layer_na(self, emsr_code: str, folder_name: str, layer: str):
        """Mark a specific layer as NA in the tracking CSV."""
        rec = self.get(folder_name)
        if rec:
            rec[layer] = "NA"
        else:
            # Create new record with NA
            rec = {
                'EMSR_code': emsr_code,
                'folder_name': folder_name,
                'S1': 'no',
                'S2_indices': 'no',
                'MERIT': 'no',
                'Soil': 'no',
                'ESA_PW': 'no',
                'Precipitation': 'no',
                'SoilMoisture': 'no',
            }
            rec[layer] = 'NA'
        self.upsert(rec)

    def _flush(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ['EMSR_code', 'folder_name', 'S1', 'S2_indices',
                         'MERIT', 'Soil', 'ESA_PW', 'Precipitation', 'SoilMoisture']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._data.values():
                writer.writerow(row)


# ─── UTILITIES ───────────────────────────────────────────────────────────────

def _parse_event_date(act_folder_name: str) -> Optional[date]:
    """Extract YYYYMMDD date from standardized folder name suffix."""
    m = re.search(r"_(\d{8})$", act_folder_name)
    if not m:
        return None
    try:
        return date(int(m.group(1)[:4]),
                    int(m.group(1)[4:6]),
                    int(m.group(1)[6:8]))
    except ValueError:
        return None


def _read_aoi_bounds(act_folder: Path) -> Optional[Tuple[float, float, float, float]]:
    """Return (minx, miny, maxx, maxy) in WGS84 from AOI shapefile."""
    aoi_shp = act_folder / "aoi" / "aoi.shp"
    if not aoi_shp.exists():
        return None
    try:
        gdf = gpd.read_file(aoi_shp)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return tuple(gdf.total_bounds)   # (minx, miny, maxx, maxy)
    except Exception as e:
        print(f"      ! could not read AOI shapefile: {e}")
        return None


def _buffer_bounds(minx: float, miny: float, maxx: float, maxy: float,
                   buffer_deg: float) -> Tuple[float, float, float, float]:
    """Expand a lon/lat bbox by buffer_deg on every side (clamped to valid range)."""
    return (
        max(minx - buffer_deg, -180.0),
        max(miny - buffer_deg,  -90.0),
        min(maxx + buffer_deg,  180.0),
        min(maxy + buffer_deg,   90.0),
    )


def _snap_bounds(minx: float, miny: float, maxx: float, maxy: float,
                 pixel: float) -> Tuple[float, float, float, float, list]:
    """
    Snap bounds outward to the nearest pixel_deg grid and return the
    GEE crsTransform list [xScale, xShear, xOrigin, yShear, yScale, yOrigin].
    """
    snapped_minx = math.floor(minx / pixel) * pixel
    snapped_miny = math.floor(miny / pixel) * pixel
    snapped_maxx = math.ceil(maxx  / pixel) * pixel
    snapped_maxy = math.ceil(maxy  / pixel) * pixel
    # affine: [x_scale, x_rot, x_origin, y_rot, y_scale, y_origin]
    crs_transform = [pixel, 0, snapped_minx, 0, -pixel, snapped_maxy]
    return snapped_minx, snapped_miny, snapped_maxx, snapped_maxy, crs_transform


def _gee_region(minx: float, miny: float, maxx: float, maxy: float):
    """Build an ee.Geometry rectangle."""
    return ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:4326",
                                  evenOdd=True)


def _submit(image, description: str, file_prefix: str,
            region, crs_transform: list, act_name: str, layer: str) -> bool:
    """
    Submit one GEE export task.
    Returns True on success.
    """
    # GEE description must be ≤100 chars and unique-ish
    short_desc = description[:100]
    # Each activation gets its own folder in Google Drive
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=short_desc,
        folder=act_name,  # Use activation folder name instead of single shared folder
        fileNamePrefix=file_prefix,
        crs="EPSG:4326",
        crsTransform=crs_transform,
        region=region,
        maxPixels=int(1e13),
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": False},
    )
    try:
        task.start()
        task_id = task.id
        print(f"        ✓ Submitted task {task_id}")
        return True
    except Exception as e:
        print(f"        ✗ Submission failed: {e}")
        return False


# ─── COVERAGE VALIDATION ─────────────────────────────────────────────────────

def is_valid_tif(file_path: Path, expected_bands: int) -> bool:
    """
    Check if a TIF file is valid (non-zero size, readable, correct band count).
    Returns False for corrupted files (0-byte files, wrong band count, etc.).
    """
    if not file_path.exists():
        return False

    # Check file size (0-byte files are corrupted)
    if file_path.stat().st_size == 0:
        return False

    try:
        with rasterio.open(file_path) as src:
            # Check band count
            if src.count != expected_bands:
                return False
            # Try to read metadata (validates file structure)
            _ = src.bounds
            _ = src.transform
            return True
    except Exception:
        return False


def check_layer_coverage(raster_path: Path, flood_extent_shp: Path) -> bool:
    """
    Check if raster spatially overlaps with flood extent geometry.

    Transforms both raster and flood extent to same CRS before comparison.

    Returns True if overlap exists, False otherwise.
    """
    if not raster_path.exists():
        return False

    if not flood_extent_shp.exists():
        return False

    try:
        # Read flood extent shapefile
        flood_gdf = gpd.read_file(flood_extent_shp)
        if not flood_gdf.crs:
            return False

        # Open raster to get its CRS
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            raster_bounds = src.bounds  # (left, bottom, right, top)

            if not raster_crs:
                return False

            # Transform flood extent to raster CRS for comparison
            flood_gdf_transformed = flood_gdf.to_crs(raster_crs)
            flood_bounds = flood_gdf_transformed.total_bounds  # (minx, miny, maxx, maxy)

            # Check if bounds overlap (simple bbox intersection test)
            # No overlap if: raster is entirely left/right/above/below flood extent
            if (raster_bounds.right < flood_bounds[0] or   # raster entirely west of flood
                raster_bounds.left > flood_bounds[2] or    # raster entirely east of flood
                raster_bounds.top < flood_bounds[1] or     # raster entirely south of flood
                raster_bounds.bottom > flood_bounds[3]):   # raster entirely north of flood
                return False

            return True

    except Exception as e:
        print(f"        ! Coverage check failed for {raster_path.name}: {e}")
        return False


def find_layer_in_exports(folder_name: str, layer_name: str) -> Optional[Path]:
    """
    Find a layer file in GEE_exports/{folder_name}/.

    Args:
        folder_name: Full activation folder name
        layer_name: Layer filename (e.g., "S1_VV_VH.tif", "land_cover.tif")

    Returns:
        Path to layer file if found, None otherwise
    """
    search_dir = GEE_EXPORTS_DIR / folder_name
    if search_dir.exists():
        layer_path = search_dir / layer_name
        if layer_path.exists():
            return layer_path
    return None


def update_download_tracking_csv(download_tracker: DownloadTracker):
    """
    Scan activations_reorganized and GEE_exports to update the export-status CSV.

    Creates:
        - the per-layer GEE export-status CSV (config.CSV_GEE_EXPORT_STATUS),
          one row per activation that has a flood-extent polygon.

    Only processes activations from 2017 onwards.

    Status values:
        - "NA" = No GEE images available (marked during processing)
        - "no" = File not yet available in exports
        - "yes" = File exists and passes validation
    """
    print("\nUpdating download tracking CSV …")

    if not ACTIVATIONS_DIR.exists():
        print(f"  ! activations dir not found: {ACTIVATIONS_DIR}")
        return

    tracking_records = []

    # standardized structure: activations_reorganized/{EMSR_CODE}/{activation_folder}/
    activation_folders = sorted(
        sub for emsr_dir in ACTIVATIONS_DIR.iterdir() if emsr_dir.is_dir()
        for sub in emsr_dir.iterdir() if sub.is_dir()
    )

    if not activation_folders:
        print(f"  ! No activation folders found in {ACTIVATIONS_DIR}")
        return

    print(f"  Scanning {len(activation_folders)} activation folders …")

    for act_folder in activation_folders:
            folder_name = act_folder.name
            emsr_code = folder_name.split("_")[0] if "_" in folder_name else folder_name

            # Skip activations before 2017
            event_date = _parse_event_date(folder_name)
            if event_date and event_date.year < 2017:
                continue

            flood_extent_shp = act_folder / "flood_extent" / "event.shp"

            # Skip activations without a flood-extent polygon.
            if not flood_extent_shp.exists():
                continue

            # Get existing record to preserve NA statuses
            existing_rec = download_tracker.get(folder_name)

            # Check each layer
            record = {
                'EMSR_code': emsr_code,
                'folder_name': folder_name,
            }

            # Layers to check come from the enabled registry: composite layers
            # (S1, S2 indices) additionally get an AOI-coverage check.
            layers_to_check = [
                (spec.key, spec.filename, spec.band_count(),
                 spec.kind == "composite")
                for spec in enabled_layers()
            ]

            for layer_key, layer_file, expected_bands, needs_coverage in layers_to_check:
                # Preserve NA status if it was already marked
                if existing_rec and existing_rec.get(layer_key) == "NA":
                    record[layer_key] = 'NA'
                else:
                    # Check if file exists in exports
                    layer_path = find_layer_in_exports(folder_name, layer_file)

                    if layer_path is None:
                        record[layer_key] = 'no'
                    else:
                        # First validate band count
                        if not is_valid_tif(layer_path, expected_bands):
                            record[layer_key] = 'no'  # Invalid/corrupted file
                        elif needs_coverage:
                            # S1 or S2_indices: check coverage
                            has_coverage = check_layer_coverage(layer_path, flood_extent_shp)
                            record[layer_key] = 'yes' if has_coverage else 'no'
                        else:
                            # Other layers: band count already validated
                            record[layer_key] = 'yes'

            tracking_records.append(record)

    # Write download tracking CSV
    META_DIR.mkdir(parents=True, exist_ok=True)

    if tracking_records:
        with open(GEE_TASKS_CSV, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['EMSR_code', 'folder_name'] + ALL_LAYERS
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(tracking_records)

        print(f"  ✓ Written {len(tracking_records)} records to {GEE_TASKS_CSV}")
    else:
        print(f"  ! No activations with flood extent found")


# ─── GEE IMAGE BUILDERS ───────────────────────────────────────────────────────

def calculate_coverage(image, aoi):
    """
    Calculate percentage of AOI covered by valid (non-masked) pixels.

    Uses multi-scale approach (10m → 30m → 100m) to handle large AOIs.
    Returns actual coverage percentage as a float.

    This is a CLIENT-SIDE check using .getInfo() to retrieve the value.
    """
    # Get validity mask (min of all band masks)
    valid_mask = image.mask().reduce(ee.Reducer.min()).rename('mask')

    # Calculate total AOI area
    aoi_area = aoi.area(maxError=1)

    # Try multiple scales (coarse to fine) to avoid memory errors
    for scale in [100, 30, 10]:
        try:
            # Sum valid pixel areas
            result = valid_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi,
                scale=scale,
                maxPixels=1e13,
                bestEffort=True
            )

            covered_area_raw = result.get('mask')
            if covered_area_raw is None:
                continue

            covered_area = ee.Number(covered_area_raw)
            coverage = covered_area.divide(aoi_area).multiply(100)

            # CLIENT-SIDE: Retrieve actual value from server
            coverage_value = coverage.getInfo()
            return coverage_value

        except Exception as e:
            if scale == 100:  # Last attempt failed
                print(f"        ! Coverage calculation failed at all scales: {e}")
                return 0.0
            continue  # Try coarser scale

    return 0.0


def build_s1(region, event_date: date):
    """
    Sentinel-1 IW VV/VH median composite with iterative coverage checking.

    Strategy - Smart Iterative Approach:
      1. Iterate through temporal windows: 15d → 30d → 45d → 60d
      2. For each window:
         a. Build composite
         b. Calculate ACTUAL coverage using .getInfo()
         c. Print real coverage percentage
         d. If coverage >= threshold (99%): STOP and use this composite
         e. If coverage < threshold: continue to next temporal window

    Why This is SMARTER:
      ✓ Uses .getInfo() to measure ACTUAL coverage
      ✓ Stops early when coverage sufficient (doesn't waste computation)
      ✓ Prints real coverage percentages (user visibility)
      ✓ Only builds composites that are needed
      ✓ Prioritizes recent data (15d) over older data (60d)

    Returns 2-band image: VV, VH.
    """
    def _get_s1_collection(window_days: int):
        """Get S1 collection for given window."""
        start = (event_date - timedelta(days=window_days)).isoformat()
        end   = event_date.isoformat()
        return (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select(["VV", "VH"])
        )

    print(f"        S1 Windows: {S1_WINDOWS} days, target coverage: {S1_TARGET_COVERAGE*100}%")
    print(f"        S1: Iterating through temporal windows...")

    s1_result = None
    s1_coverage = 0.0

    for window_days in S1_WINDOWS:
        print(f"        S1   → Window {window_days} days: ", end="")

        col = _get_s1_collection(window_days)
        composite = col.median()

        # Measure ACTUAL coverage using .getInfo()
        try:
            coverage = calculate_coverage(composite, region)
            print(f"Coverage = {coverage:.2f}%")

            # Progressive gap filling: keep best result so far
            if s1_result is None:
                s1_result = composite
            else:
                s1_result = s1_result.unmask(composite)

            s1_coverage = coverage

            # STOP if coverage sufficient
            if coverage >= (S1_TARGET_COVERAGE * 100):
                print(f"        S1   ✓ Achieved {coverage:.2f}% coverage (threshold: {S1_TARGET_COVERAGE*100}%)")
                print(f"        S1   ✓ STOPPING - Coverage sufficient!")
                break

        except Exception as e:
            print(f"Failed to calculate coverage: {e}")
            # Keep building composites even if coverage check fails
            if s1_result is None:
                s1_result = composite
            else:
                s1_result = s1_result.unmask(composite)

    # If no composite was built (no S1 images available), mark as NA
    if s1_result is None:
        print(f"        S1   ! No S1 images available - SKIPPING task submission")
        return None

    # Check if final coverage is sufficient
    if s1_coverage < (S1_TARGET_COVERAGE * 100):
        print(f"        S1   ! Coverage {s1_coverage:.2f}% < {S1_TARGET_COVERAGE*100}% after all windows - SKIPPING task submission")
        return None

    return s1_result.unmask(-9999)


def build_s2_indices(region, event_date: date):
    """
    Sentinel-2 NDVI/NDBI composite with iterative coverage checking.

    Strategy - Smart Iterative Approach:
      1. Current Year (S2_WINDOW_DAYS before event):
         - Iterate through cloud thresholds: initial → +10% → +10% → ...
         - For each threshold:
           a. Build composite
           b. Calculate ACTUAL coverage using .getInfo()
           c. Print real coverage percentage
           d. If coverage >= threshold (98%): STOP and use this composite
           e. If coverage < threshold: continue to next cloud threshold

      2. Seasonal Fallback (only if all current year thresholds insufficient):
         - Same iterative approach with seasonal period
         - 2017-2024: Use next year (more data available)
         - 2025+: Use previous year (recent events)

    Why This is SMARTER:
      ✓ Uses .getInfo() to measure ACTUAL coverage (not server-side blind check)
      ✓ Stops early when coverage sufficient (doesn't waste computation)
      ✓ Prints real coverage percentages (user visibility)
      ✓ Only builds composites that are needed
      ✓ Learned from proven production code (download_gee_layers.py)

    Returns 2-band image: NDVI, NDBI (no date information in bands).
    """
    print(f"        S2 Config: window={S2_WINDOW_DAYS}d, cloud_initial={S2_CLOUD_INITIAL}%, "
          f"cloud_progressive={S2_CLOUD_PROGRESSIVE}, coverage_threshold={S2_COVERAGE_THRESHOLD}")
    print(f"        S2 Period: {(event_date - timedelta(days=S2_WINDOW_DAYS)).isoformat()} to {event_date.isoformat()}")

    def _get_s2_collection(start_date: date, end_date: date, cloud_threshold: int):
        """Get S2 collection for date range, filtered by cloud threshold."""
        return (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date.isoformat(), end_date.isoformat())
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        )

    def _build_indices_from_collection(col):
        """Build NDVI/NDBI median composite from collection."""
        s2_median = col.median()
        ndvi = s2_median.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndbi = s2_median.normalizedDifference(["B11", "B8"]).rename("NDBI")
        return ee.Image.cat([ndvi, ndbi])

    # ── Iterative Coverage Check: Current Year ────────────────────────────
    start_date = event_date - timedelta(days=S2_WINDOW_DAYS)
    end_date = event_date
    all_cloud_thresholds = [S2_CLOUD_INITIAL] + S2_CLOUD_PROGRESSIVE

    print(f"        S2 Current Year: Iterating through cloud thresholds...")

    current_year_result = None
    current_year_coverage = 0.0

    for cloud_threshold in all_cloud_thresholds:
        print(f"        S2   → Cloud threshold {cloud_threshold}%: ", end="")

        col = _get_s2_collection(start_date, end_date, cloud_threshold)

        # Check if collection has images BEFORE building composite
        try:
            num_images = col.size().getInfo()
            if num_images == 0:
                print(f"No S2 images (0 images found)")
                continue  # Skip to next cloud threshold
        except Exception as e:
            print(f"Failed to check collection size: {e}")
            continue

        print(f"{num_images} images, ", end="")
        composite = _build_indices_from_collection(col)

        # Measure ACTUAL coverage using .getInfo()
        try:
            coverage = calculate_coverage(composite, region)
            print(f"Coverage = {coverage:.2f}%")

            # Progressive gap filling: keep best result so far
            if current_year_result is None:
                current_year_result = composite
            else:
                current_year_result = current_year_result.unmask(composite)

            current_year_coverage = coverage

            # STOP if coverage sufficient
            if coverage >= (S2_COVERAGE_THRESHOLD * 100):
                print(f"        S2   ✓ Achieved {coverage:.2f}% coverage (threshold: {S2_COVERAGE_THRESHOLD*100}%)")
                print(f"        S2   ✓ STOPPING - Coverage sufficient!")
                break

        except Exception as e:
            print(f"Failed to calculate coverage: {e}")
            # Keep building composites even if coverage check fails
            if current_year_result is None:
                current_year_result = composite
            else:
                current_year_result = current_year_result.unmask(composite)

    # If no composite was built (no S2 images available), mark as NA
    if current_year_result is None:
        print(f"        S2   ! No S2 images available - SKIPPING task submission")
        return None

    # ── Check if Seasonal Fallback Needed ─────────────────────────────────
    if not S2_USE_SEASONAL_FALLBACK:
        print(f"        S2 Seasonal fallback: DISABLED")
        return current_year_result.unmask(-9999)

    # Only add seasonal if coverage insufficient
    if current_year_coverage >= (S2_COVERAGE_THRESHOLD * 100):
        print(f"        S2 Seasonal fallback: NOT NEEDED (coverage {current_year_coverage:.2f}% >= {S2_COVERAGE_THRESHOLD*100}%)")
        return current_year_result.unmask(-9999)

    # Coverage insufficient - add seasonal fallback
    print(f"        S2 Seasonal fallback: NEEDED (coverage {current_year_coverage:.2f}% < {S2_COVERAGE_THRESHOLD*100}%)")

    # Determine seasonal year
    event_year = event_date.year
    if event_year <= 2024:
        seasonal_year = event_year + 1
        seasonal_strategy = "NEXT year"
    else:
        seasonal_year = event_year - 1
        seasonal_strategy = "PREVIOUS year"

    print(f"        S2 Seasonal fallback: Using {seasonal_strategy} ({seasonal_year})")

    # Build seasonal composite iteratively
    try:
        event_seasonal = date(seasonal_year, event_date.month, event_date.day)
        start_seasonal = event_seasonal - timedelta(days=S2_WINDOW_DAYS)
        end_seasonal = event_seasonal

        print(f"        S2 Seasonal period: {start_seasonal.isoformat()} to {end_seasonal.isoformat()}")
        print(f"        S2 Seasonal: Iterating through cloud thresholds...")

        seasonal_result = None
        seasonal_coverage = 0.0

        for cloud_threshold in all_cloud_thresholds:
            print(f"        S2   → Cloud threshold {cloud_threshold}%: ", end="")

            col = _get_s2_collection(start_seasonal, end_seasonal, cloud_threshold)

            # Check if collection has images BEFORE building composite
            try:
                num_images = col.size().getInfo()
                if num_images == 0:
                    print(f"No S2 images (0 images found)")
                    continue  # Skip to next cloud threshold
            except Exception as e:
                print(f"Failed to check collection size: {e}")
                continue

            print(f"{num_images} images, ", end="")
            composite = _build_indices_from_collection(col)

            # Measure ACTUAL coverage
            try:
                coverage = calculate_coverage(composite, region)
                print(f"Coverage = {coverage:.2f}%")

                # Progressive gap filling
                if seasonal_result is None:
                    seasonal_result = composite
                else:
                    seasonal_result = seasonal_result.unmask(composite)

                seasonal_coverage = coverage

                # STOP if coverage sufficient
                if coverage >= (S2_COVERAGE_THRESHOLD * 100):
                    print(f"        S2   ✓ Seasonal achieved {coverage:.2f}% coverage")
                    print(f"        S2   ✓ STOPPING - Coverage sufficient!")
                    break

            except Exception as e:
                print(f"Failed to calculate coverage: {e}")
                if seasonal_result is None:
                    seasonal_result = composite
                else:
                    seasonal_result = seasonal_result.unmask(composite)

        # Fill current year gaps with seasonal
        if seasonal_result is not None:
            final_result = current_year_result.unmask(seasonal_result)

            # Calculate final coverage after adding seasonal
            try:
                final_coverage = calculate_coverage(final_result, region)
                print(f"        S2   ✓ Added seasonal fallback, final coverage = {final_coverage:.2f}%")

                if final_coverage < (S2_COVERAGE_THRESHOLD * 100):
                    print(f"        S2   ! Coverage {final_coverage:.2f}% < {S2_COVERAGE_THRESHOLD*100}% even after seasonal - SKIPPING task submission")
                    return None

                return final_result.unmask(-9999)
            except Exception as e:
                print(f"        S2   ! Failed to calculate final coverage: {e} - SKIPPING task submission")
                return None
        else:
            print(f"        S2   ! No seasonal images available, coverage {current_year_coverage:.2f}% < {S2_COVERAGE_THRESHOLD*100}% - SKIPPING task submission")
            return None

    except ValueError:
        # Handle leap year edge case (Feb 29)
        print(f"        S2 Seasonal fallback: Leap year edge case (Feb 29) - skipping seasonal")
        if current_year_coverage < (S2_COVERAGE_THRESHOLD * 100):
            print(f"        S2   ! Coverage {current_year_coverage:.2f}% < {S2_COVERAGE_THRESHOLD*100}% - SKIPPING task submission")
            return None
        return current_year_result.unmask(-9999)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def submit_for_activation(act_folder: Path, download_tracker: DownloadTracker) -> int:
    """
    Submit all missing GEE tasks for one activation folder.
    Returns number of new tasks submitted.
    """
    act_name = act_folder.name

    # Extract EMSR code from folder name (e.g. EMSR773_AOI01_... → EMSR773)
    emsr_code = act_name.split("_")[0] if "_" in act_name else act_name

    # Check if activation has any NA layers - skip if so
    if download_tracker.has_na(act_name):
        print(f"    ⚠ Activation has NA layers (no images available) — skipping")
        return 0

    # Parse event date
    event_date = _parse_event_date(act_name)
    if not event_date:
        print(f"    ! cannot parse event date from '{act_name}' — skipping")
        return 0

    # Read AOI bounds
    bounds = _read_aoi_bounds(act_folder)
    if not bounds:
        print(f"    ! no AOI shapefile found — skipping")
        return 0

    minx, miny, maxx, maxy, crs_tf = _snap_bounds(*bounds, PIXEL_DEG)
    region = _gee_region(minx, miny, maxx, maxy)

    submitted = 0

    # Helper: build Drive file prefix (creates subfolder in Drive)
    def prefix(layer: str) -> str:
        return f"{act_name}/{layer}"

    def desc(layer: str) -> str:
        import unicodedata, re as _re
        safe = unicodedata.normalize("NFKD", act_name).encode("ascii", "ignore").decode("ascii")
        safe = _re.sub(r"[^a-zA-Z0-9._,:;\-]", "_", safe)
        return f"{safe[:70]}_{layer}"

    # Resolution (degrees) per layer key for the temporal layers' own grid.
    TEMPORAL_DEG = {"Precipitation": PRECIP_DEG, "SoilMoisture": SMAP_DEG}
    # Buffered AOI bbox for temporal layers, so every patch later has a full
    # spatial neighbourhood of coarse weather pixels around it. The buffer width
    # is a user knob (config.TEMPORAL_BUFFER_DEG); widen it to capture upstream
    # rainfall and soil moisture that drive downstream discharge.
    buf = _buffer_bounds(*bounds, config.TEMPORAL_BUFFER_DEG)

    # Composite builders (S1, S2 indices) live in this script; refer to them by
    # the builder_name carried in the registry.
    COMPOSITE_BUILDERS = {"build_s1": build_s1, "build_s2_indices": build_s2_indices}

    # Submit one task per enabled layer, dispatched by kind.
    for spec in enabled_layers():
        key = spec.key
        if not download_tracker.needs_submission(act_name, key):
            continue
        print(f"      → {key}")

        if spec.kind == "composite":
            build = COMPOSITE_BUILDERS[spec.builder_name]
            img = build(region, event_date)
            if img is None:
                print(f"        ✗ Marked as NA (no images available)")
                download_tracker.mark_layer_na(emsr_code, act_name, key)
                continue
            ok = _submit(img, desc(key), prefix(key), region, crs_tf, act_name, key)

        elif spec.kind == "temporal":
            deg = TEMPORAL_DEG.get(key, spec.resolution_m / 111000.0)
            t_minx, t_miny, t_maxx, t_maxy, t_tf = _snap_bounds(*buf, deg)
            t_region = _gee_region(t_minx, t_miny, t_maxx, t_maxy)
            img = spec.builder(t_region, event_date, spec.n_days)
            # Stamp the antecedent window into the filename: oldest day (event
            # date minus n_days) and newest day (event date minus 1, event day
            # excluded), e.g. Precipitation_20200714_20200812.tif.
            first_day = (event_date - timedelta(days=spec.n_days)).strftime("%Y%m%d")
            last_day  = (event_date - timedelta(days=1)).strftime("%Y%m%d")
            dated = f"{spec.filename[:-4]}_{first_day}_{last_day}"
            ok = _submit(img, desc(dated), prefix(dated), t_region, t_tf, act_name, key)

        else:  # static
            img = spec.builder(region)
            # Export each static layer on its own native-resolution grid over the
            # same AOI footprint. ESA_PW (10 m) lands on the fine reference grid;
            # MERIT (90 m) and SoilGrids (250 m) keep their coarse native pixels.
            if abs(spec.resolution_m - 10.0) < 1e-6:
                ok = _submit(img, desc(key), prefix(key), region, crs_tf, act_name, key)
            else:
                deg = spec.resolution_m / 111000.0
                s_minx, s_miny, s_maxx, s_maxy, s_tf = _snap_bounds(*bounds, deg)
                s_region = _gee_region(s_minx, s_miny, s_maxx, s_maxy)
                ok = _submit(img, desc(key), prefix(key), s_region, s_tf, act_name, key)

        submitted += int(ok)
        time.sleep(REQUEST_DELAY)

    return submitted


def main():
    parser = argparse.ArgumentParser(description="Submit GEE export tasks for flood activations")
    parser.add_argument('--update-tracking', action='store_true',
                       help='Only update download tracking CSV (no submissions)')
    args = parser.parse_args()

    submit_to_gee = (SUBMIT_TO_GEE.lower() == 'yes')
    status_only = args.update_tracking

    print("=" * 72)
    print("  GEE Export Task Submission  (Script 2)")
    print(f"  BASE_DIR         : {BASE_DIR}")
    print(f"  Activations dir  : {ACTIVATIONS_DIR}")
    print(f"  GEE exports dir  : {GEE_EXPORTS_DIR}")
    if status_only:
        print(f"  Mode             : UPDATE TRACKING ONLY (--update-tracking)")
    elif not submit_to_gee:
        print(f"  Mode             : TRACKING ONLY (submit_to_gee=no)")
    else:
        print(f"  Mode             : SUBMIT TO GEE (submit_to_gee=yes)")
    print(f"  Layers enabled   : {', '.join(ALL_LAYERS)}")
    print("=" * 72)

    config.migrate_csv_names()  # rename any old-named metadata files in place

    # ── Authenticate GEE ─────────────────────────────────────────────────
    if submit_to_gee and not status_only:
        print("\nInitialising GEE …")
        try:
            ee.Initialize()
            print("  ✓ GEE authenticated")
        except Exception as e:
            print(f"  ✗ GEE init failed: {e}")
            print("  Run:  earthengine authenticate")
            sys.exit(1)

    META_DIR.mkdir(parents=True, exist_ok=True)

    # ── Initialize Download Tracker ──────────────────────────────────────
    download_tracker = DownloadTracker(GEE_TASKS_CSV)

    # ── Status Check Mode ────────────────────────────────────────────────
    if status_only:
        # Update tracking CSV and exit
        update_download_tracking_csv(download_tracker)
        return

    # ── GEE Task Submission ───────────────────────────────────────────────
    if submit_to_gee:

        # ── Find activation folders ───────────────────────────────────────
        if not ACTIVATIONS_DIR.exists():
            print(f"\n! activations dir not found: {ACTIVATIONS_DIR}")
            print("  Run Script 1 first to download + convert activations.")
            sys.exit(1)

        # standardized structure: activations_reorganized/{EMSR_CODE}/{activation_folder}/
        # Collect all activation subfolders (one level below EMSR parent)
        folders = sorted(
            sub for emsr_dir in ACTIVATIONS_DIR.iterdir() if emsr_dir.is_dir()
            for sub in emsr_dir.iterdir() if sub.is_dir()
        )
        if not folders:
            print(f"\n! No activation folders in {ACTIVATIONS_DIR}")
            sys.exit(1)

        print(f"\nFound {len(folders)} activation folders")

        if TEST_MODE:
            print(f"  TEST_MODE enabled: Processing ONLY first valid activation folder (year >= 2017)")

        total_submitted = 0
        total_skipped   = 0
        total_skipped_na = 0
        total_no_date   = 0
        processed_count = 0

        for i, folder in enumerate(folders, 1):
            act_name = folder.name

            # Skip activations before 2017
            event_date = _parse_event_date(act_name)
            if event_date and event_date.year < 2017:
                continue

            # Skip activations with any NA layers
            if download_tracker.has_na(act_name):
                total_skipped_na += 1
                continue

            # Count how many layers still need submission
            missing = [l for l in ALL_LAYERS
                       if download_tracker.needs_submission(act_name, l)]

            if not missing:
                total_skipped += 1
                if TEST_MODE and processed_count > 0:
                    print(f"  TEST_MODE: Stopping after processing first activation")
                    break
                continue

            print(f"\n[{i}/{len(folders)}] {act_name}")
            print(f"    Missing layers: {', '.join(missing)}")

            n = submit_for_activation(folder, download_tracker)
            total_submitted += n
            if n == 0 and missing:
                total_no_date += 1

            processed_count += 1

            # In TEST_MODE, stop after processing first valid activation
            if TEST_MODE:
                print(f"\n  TEST_MODE: Processed first activation folder. Stopping.")
                break

        # ── Summary ───────────────────────────────────────────────────────────
        print()
        print("=" * 72)
        print("  SUMMARY")
        print("=" * 72)
        print(f"  Activations fully submitted       : {total_skipped}")
        print(f"  Activations skipped (has NA layers): {total_skipped_na}")
        print(f"  New tasks submitted                : {total_submitted}")
        print(f"  Skipped (no date / no AOI)         : {total_no_date}")
        print()
        print(f"  Download tracking CSV: {GEE_TASKS_CSV}")
        print()
        print("  Note: Activations with 'NA' status in any layer are skipped")
        print("        (NA = no GEE images available for that layer)")
        print()
        print("  Next steps after GEE tasks complete:")
        print(f"    1. Download activation folders from Google Drive")
        print(f"       Each activation has its own folder (e.g., EMSR123_AOI01_...)")
        print(f"       Inside each folder: S1_VV_VH.tif, land_cover.tif, MERIT.tif, etc.")
        print(f"    2. Run Script 4 to validate exports, then Script 5 to process downloads → patches")
        print("=" * 72)

    # ── Update Download Tracking CSV ──────────────────────────────────────
    update_download_tracking_csv(download_tracker)

    if not submit_to_gee:
        print()
        print("=" * 72)
        print("  Download tracking CSV updated (GEE submission skipped)")
        print(f"  Tracking CSV: {GEE_TASKS_CSV}")
        print("=" * 72)


if __name__ == "__main__":
    main()
