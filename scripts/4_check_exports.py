#!/usr/bin/env python3
"""
Script 4: Validate GEE Exports and Build flood_dataset.csv

Scans all GEE export folders and validates that each activation has all required
layers with correct band counts. Merges sensor info from activations.csv (Script 1
output) and adds resolution metadata. Produces the final shareable dataset CSV.

Output files:
  data/flood_dataset.csv           — complete activations with full metadata
  metadata/flood_dataset.csv       — copy for archival

Required layers per activation:
  S1_VV_VH.tif        2 bands
  land_cover.tif      2 bands
  MERIT.tif           4 bands
  Soil.tif            2 bands
  ESA_PW.tif          1 band
  Precipitation.tif  10 bands
  SoilMoisture.tif   10 bands

Usage:
  python scripts/4_validate_exports.py
"""

import csv
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not found. Install with: pip install rasterio")
    sys.exit(1)

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    print("ERROR: geopandas not found. Install with: pip install geopandas")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests not found. Install with: pip install requests")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not found. Install with: pip install pandas")
    sys.exit(1)


# ─── PATH SETUP ──────────────────────────────────────────────────────────────

BASE_DIR             = Path(__file__).resolve().parent.parent
DATA_DIR             = BASE_DIR / "data"
META_DIR             = BASE_DIR / "metadata"
GEE_EXPORTS_DIR      = DATA_DIR / "GEE_exports"
DCC_ACTIVATIONS_DIR  = DATA_DIR / "activations" / "activations_dcc"
ACTIVATIONS_CSV      = DATA_DIR / "activations.csv"        # Script 1 output
GEE_TASKS_CSV         = META_DIR / "gee_tasks_record.csv"
FLOOD_DATASET_CSV    = DATA_DIR / "flood_dataset.csv"      # primary output
FLOOD_DATASET_META   = META_DIR / "flood_dataset.csv"      # archival copy
HYDROBASINS_DIR      = DATA_DIR / "hydrobasins"

# Required layers with expected band counts
REQUIRED_LAYERS = {
    "S1_VV_VH": {"files": ["S1_VV_VH.tif"], "bands": 2},
    "land_cover": {"files": ["land_cover.tif"], "bands": 2},
    "MERIT": {"files": ["MERIT.tif"], "bands": 4},
    "Soil": {"files": ["Soil.tif"], "bands": 2},
    "ESA_PW": {"files": ["ESA_PW.tif", "ESA_WorldCover_PermanentWater.tif"], "bands": 1},
    "Precipitation": {"files": ["Precipitation.tif"], "bands": 10},
    "SoilMoisture": {"files": ["SoilMoisture.tif"], "bands": 10},
}


# ─── RESOLUTION MAPPING ──────────────────────────────────────────────────────
# Actual resolutions extracted from 930 Europe activation PDFs (not theoretical specs).
# Flood mapping uses wide-swath modes — values reflect real usage.

SENSOR_RESOLUTION = {
    'Aerial': 0.2, 'Plane imagery': 0.2, 'UAS/UAV': 0.5,
    'Pléiades Neo': 0.3,
    'Pléiades-1A': 0.5, 'Pléiades-1B': 0.5, 'Pléiades-1A/B': 0.5, 'Pléiades': 0.5,
    'WorldView-1': 0.5, 'WorldView-2': 0.5, 'WorldView-3': 0.5, 'WorldView-4': 0.5,
    'GeoEye-1': 0.5, 'SkySat': 0.5, 'Legion': 0.5, 'Deimos-2': 0.75,
    'SPOT-6': 1.5, 'SPOT-7': 1.5, 'SPOT-6/7': 1.5, 'SPOT': 1.5,
    'ICEYE': 2.5,
    'TerraSAR-X': 3.0, 'COSMO-SkyMed SG': 3.0, 'PlanetScope': 3.0,
    'ALOS-2': 3.0, 'RADARSAT Constellation': 3.0, 'orthoimages': 3.9,
    'RADARSAT-2': 4.0, 'COSMO-SkyMed': 5.0, 'RapidEye': 5.0,
    'Sentinel-1A': 10.0, 'Sentinel-1B': 10.0, 'Sentinel-1A/B': 10.0, 'Sentinel-1': 10.0,
    'Sentinel-2A': 10.0, 'Sentinel-2B': 10.0, 'Sentinel-2A/B': 10.0,
    'Sentinel-2': 10.0, 'Sentinel': 10.0, 'SAOCOM': 10.0,
    'PAZ': 15.0, 'Landsat-8': 15.0, 'Landsat-9': 15.0,
    'ESRI World Imagery': 30.0,
}


def get_post_sensor_resolution(post_sensor_string: str) -> Optional[float]:
    """Return best (minimum) resolution from a comma/semicolon-separated sensor string."""
    import re as _re
    if not post_sensor_string or not post_sensor_string.strip():
        return None
    resolutions = [
        SENSOR_RESOLUTION[s.strip()]
        for s in _re.split('[,;]', post_sensor_string)
        if s.strip() in SENSOR_RESOLUTION
    ]
    return min(resolutions) if resolutions else None


def classify_resolution(resolution_m: Optional[float]) -> str:
    if resolution_m is None:
        return 'unknown'
    if resolution_m < 3.0:
        return 'very-high'
    if resolution_m < 10.0:
        return 'high'
    return 'medium'


def detect_region(aoi_shp_path: Path) -> str:
    """Determine 'europe' or 'rest_of_world' from AOI centroid coordinates."""
    try:
        gdf = gpd.read_file(aoi_shp_path)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        centroid = gdf.unary_union.centroid
        # Bounding box: Europe approx -25°W to 45°E, 34°N to 72°N
        if -25 <= centroid.x <= 45 and 34 <= centroid.y <= 72:
            return 'europe'
        return 'rest_of_world'
    except Exception:
        return 'unknown'


# ─── VALIDATION FUNCTIONS ────────────────────────────────────────────────────

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


def validate_layer(folder: Path, layer_name: str, layer_spec: Dict) -> bool:
    """
    Validate that a layer exists and has correct band count.

    Args:
        folder: Activation folder path
        layer_name: Layer name (e.g., "S1_VV_VH")
        layer_spec: Dict with "files" (list of possible filenames) and "bands" (expected count)

    Returns:
        True if valid layer found, False otherwise
    """
    expected_bands = layer_spec["bands"]
    possible_files = layer_spec["files"]

    for filename in possible_files:
        file_path = folder / filename
        if is_valid_tif(file_path, expected_bands):
            return True

    return False


def validate_activation(folder: Path) -> Tuple[bool, Dict[str, bool], List[str]]:
    """
    Validate all layers for one activation folder.

    Returns:
        Tuple of (is_complete, layer_status_dict, missing_layers_list)
    """
    layer_status = {}
    missing_layers = []

    for layer_name, layer_spec in REQUIRED_LAYERS.items():
        is_valid = validate_layer(folder, layer_name, layer_spec)
        layer_status[layer_name] = is_valid
        if not is_valid:
            missing_layers.append(layer_name)

    is_complete = len(missing_layers) == 0

    return is_complete, layer_status, missing_layers


# ─── HYDROBASINS FUNCTIONS ───────────────────────────────────────────────────

def download_hydrobasins_level12():
    """
    Download HydroSHEDS HydroBASINS level 12 global dataset.
    Downloads regional shapefiles and merges them.
    """
    HYDROBASINS_DIR.mkdir(parents=True, exist_ok=True)
    merged_file = HYDROBASINS_DIR / "hybas_lev12_global.shp"

    if merged_file.exists():
        print(f"  HydroBASINS level 12 already downloaded: {merged_file}")
        return merged_file

    print("\n  Downloading HydroBASINS level 12 data...")
    print("  This will download regional datasets from HydroSHEDS...")

    # HydroSHEDS regional downloads for level 12
    # URL pattern: https://data.hydrosheds.org/file/hydrobasins/standard/hybas_[region]_lev12_v1c.zip
    regions = [
        "af",  # Africa
        "ar",  # Arctic
        "as",  # Asia
        "au",  # Australia
        "eu",  # Europe
        "na",  # North America
        "sa",  # South America
        "si",  # Siberia
    ]

    gdfs = []

    for region in regions:
        zip_file = HYDROBASINS_DIR / f"hybas_{region}_lev12_v1c.zip"
        url = f"https://data.hydrosheds.org/file/hydrobasins/standard/hybas_{region}_lev12_v1c.zip"

        # Download if not exists
        if not zip_file.exists():
            print(f"    Downloading {region.upper()}...", end=" ", flush=True)
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()

                with open(zip_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✓")
            except Exception as e:
                print(f"✗ Failed: {e}")
                continue

        # Extract and read shapefile
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(HYDROBASINS_DIR / region)

            # Find the .shp file
            shp_files = list((HYDROBASINS_DIR / region).glob("*.shp"))
            if shp_files:
                gdf = gpd.read_file(shp_files[0])
                gdfs.append(gdf)
        except Exception as e:
            print(f"    ! Failed to extract {region}: {e}")
            continue

    if not gdfs:
        print("  ! Failed to download any HydroBASINS regions")
        return None

    # Merge all regions
    print("  Merging regional datasets...")
    global_basins = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    # Save merged file
    global_basins.to_file(merged_file)
    print(f"  ✓ Saved merged HydroBASINS to {merged_file}")

    return merged_file


def get_basin_id(aoi_shp_path: Path, basins_gdf: gpd.GeoDataFrame) -> Optional[str]:
    """
    Get HydroBASINS level 12 basin ID for an activation based on AOI centroid.

    Strategy:
    1. Try 'within' query first (centroid inside basin)
    2. If no match, find nearest basin (handles coastal/ocean areas)

    Args:
        aoi_shp_path: Path to AOI shapefile
        basins_gdf: GeoDataFrame of HydroBASINS level 12

    Returns:
        Basin ID (HYBAS_ID) as string, or None if not found
    """
    try:
        # Read AOI shapefile
        aoi_gdf = gpd.read_file(aoi_shp_path)

        # Get centroid in WGS84
        if aoi_gdf.crs.to_epsg() != 4326:
            aoi_gdf = aoi_gdf.to_crs(epsg=4326)

        centroid = aoi_gdf.unary_union.centroid

        # Ensure basins are in WGS84
        if basins_gdf.crs.to_epsg() != 4326:
            basins_gdf = basins_gdf.to_crs(epsg=4326)

        # Try 1: Find basin containing centroid
        point_gdf = gpd.GeoDataFrame({'geometry': [centroid]}, crs='EPSG:4326')
        joined = gpd.sjoin(point_gdf, basins_gdf, how='left', predicate='within')

        if len(joined) > 0 and 'HYBAS_ID' in joined.columns and not pd.isna(joined.iloc[0]['HYBAS_ID']):
            basin_id = joined.iloc[0]['HYBAS_ID']
            return str(int(basin_id))

        # Try 2: Find nearest basin (for coastal/ocean areas)
        basins_gdf['distance'] = basins_gdf.geometry.distance(centroid)
        nearest_idx = basins_gdf['distance'].idxmin()

        # Only use nearest if it's reasonably close (within ~50km = ~0.5 degrees)
        if basins_gdf.loc[nearest_idx, 'distance'] < 0.5:
            basin_id = basins_gdf.loc[nearest_idx, 'HYBAS_ID']
            return str(int(basin_id))

        return None

    except Exception as e:
        return None


# ─── MAIN ────────────────────────────────────────────────────────────────────

def find_activation_in_exports(folder_name: str) -> Optional[Path]:
    """Return path to GEE_exports/{folder_name}/ or None if not found."""
    p = GEE_EXPORTS_DIR / folder_name
    return p if p.exists() else None


def main():
    print("=" * 72)
    print("  GEE Exports Validation (Script 4)")
    print(f"  BASE_DIR              : {BASE_DIR}")
    print(f"  GEE_EXPORTS_DIR       : {GEE_EXPORTS_DIR}")
    print(f"  GEE_TASKS_CSV : {GEE_TASKS_CSV}")
    print("=" * 72)

    if not GEE_TASKS_CSV.exists():
        print(f"\n! gee_tasks_record.csv not found: {GEE_TASKS_CSV}")
        print("  Run Script 2 first (--update-tracking) to generate it.")
        sys.exit(1)

    if not GEE_EXPORTS_DIR.exists():
        print(f"\n! GEE_EXPORTS_DIR not found: {GEE_EXPORTS_DIR}")
        print("  Run Script 3 first to download GEE exports.")
        sys.exit(1)

    META_DIR.mkdir(parents=True, exist_ok=True)

    # Load HydroBASINS
    print("\n[1/4] Loading HydroBASINS level 12 data...")
    try:
        basins_file = download_hydrobasins_level12()
        if basins_file is None:
            print("  ! Failed to download HydroBASINS, basin_id will be 'unknown'")
            basins_gdf = None
        else:
            basins_gdf = gpd.read_file(basins_file)
            print(f"  ✓ Loaded {len(basins_gdf)} basins")
    except Exception as e:
        print(f"  ! Error loading HydroBASINS: {e}")
        basins_gdf = None

    # Load gee_tasks_record.csv
    print("\n[2/4] Loading gee_tasks_record.csv...")
    try:
        tracking_df = pd.read_csv(GEE_TASKS_CSV)
        print(f"  ✓ Loaded {len(tracking_df)} activations")
    except Exception as e:
        print(f"  ! Error: {e}")
        sys.exit(1)

    # Load activations.csv (sensor info from Script 1)
    print("\n[3/4] Loading activations.csv (sensor info)...")
    sensor_lookup: Dict[str, Dict] = {}
    if ACTIVATIONS_CSV.exists():
        try:
            act_df = pd.read_csv(ACTIVATIONS_CSV)
            for _, r in act_df.iterrows():
                sensor_lookup[r['folder_name']] = {
                    'pre_event_sensor': r.get('pre_event_sensor', ''),
                    'post_event_sensors': r.get('post_event_sensors', ''),
                }
            print(f"  ✓ Loaded sensor info for {len(sensor_lookup)} activations")
        except Exception as e:
            print(f"  ! Could not load activations.csv: {e} — sensor columns will be empty")
    else:
        print(f"  ! activations.csv not found at {ACTIVATIONS_CSV} — sensor columns will be empty")

    # Validate GEE exports
    print("\n[4/4] Validating GEE exports...")
    complete_records = []
    total = len(tracking_df)
    complete_count = 0

    for idx, row in tracking_df.iterrows():
        folder_name = row['folder_name']

        act_folder = find_activation_in_exports(folder_name)
        if act_folder is None:
            continue

        is_complete, _, _ = validate_activation(act_folder)
        if not is_complete:
            continue

        complete_count += 1

        # Detect region from AOI shapefile
        aoi_shp = DCC_ACTIVATIONS_DIR / folder_name / "aoi" / "aoi.shp"
        region = detect_region(aoi_shp) if aoi_shp.exists() else 'unknown'

        # Get basin ID
        basin_id = 'unknown'
        if basins_gdf is not None and aoi_shp.exists():
            result = get_basin_id(aoi_shp, basins_gdf)
            if result:
                basin_id = result

        # Sensor + resolution info
        sensors = sensor_lookup.get(folder_name, {})
        post_sensors = sensors.get('post_event_sensors', '')
        res_m = get_post_sensor_resolution(post_sensors)

        complete_records.append({
            'folder_name': folder_name,
            'region': region,
            'basin_id': basin_id,
            'pre_event_sensor': sensors.get('pre_event_sensor', ''),
            'post_event_sensors': post_sensors,
            'resolution_post_sensor': res_m if res_m is not None else '',
            'resolution_class': classify_resolution(res_m),
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{total} activations...")

    # Write flood_dataset.csv
    fieldnames = [
        'folder_name', 'region', 'basin_id',
        'pre_event_sensor', 'post_event_sensors',
        'resolution_post_sensor', 'resolution_class',
    ]
    for out_path in [FLOOD_DATASET_CSV, FLOOD_DATASET_META]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(complete_records)

    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Activations in tracking CSV : {total}")
    print(f"  Complete (all layers valid) : {complete_count}")
    print()
    print("  Output files:")
    print(f"    {FLOOD_DATASET_CSV}")
    print(f"    {FLOOD_DATASET_META}")
    print("=" * 72)


if __name__ == "__main__":
    main()
