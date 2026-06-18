#!/usr/bin/env python3
"""
Script 4: GEE Output Preprocessing — metadata builder

This step builds the dataset catalog. An event missing a CORE layer is
incomplete and is left out of the catalog; it is recorded in the missing-layers
report instead. Missing only a custom (user-added) layer is tolerated: the event
stays in the catalog and the absent custom layer is still noted in the report.

For each completed GEE export event:
  1. Reorganizes the export folder (flatten, merge GEE tiles, canonical names).
  2. Rasterizes the CEMS flood extent shapefile (flood_extent/event.shp)
     to flood_mask.tif aligned to the S1_VV_VH.tif grid (or any reference layer).
  3. Assigns HydroBASINS basin ID and region from the AOI centroid.
  4. Merges sensor metadata and writes the dataset catalog.

The set of layers checked comes from config.py (enabled layers only), so a
subset config or a custom temporal length never produces a "failed" event.

Output files:
  data/GEE_exports/{folder}/flood_mask.tif   binary flood mask per event (1=flooded)
  metadata/4_dataset_metadata.csv            final catalog (one row per cataloged event)
  metadata/4_missing_layers_report.csv       per event, which enabled layers are absent

Usage:
  python scripts/4_gee_output_preprocessing.py
"""

import csv
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.merge import merge as rio_merge
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

import config
from config import enabled_layers
from add_gee_layers import core_keys

BASE_DIR             = config.BASE_DIR
DATA_DIR             = config.DATA_DIR
META_DIR             = config.META_DIR
GEE_EXPORTS_DIR      = config.GEE_EXPORTS_DIR
DCC_ACTIVATIONS_DIR  = config.DCC_ACTIVATIONS_DIR
ACTIVATIONS_CSV      = config.CSV_ACTIVATION_CATALOG
GEE_TASKS_CSV        = config.CSV_GEE_EXPORT_STATUS
DATASET_METADATA_CSV = config.CSV_DATASET_METADATA
COMPLETE_METADATA_CSV = config.CSV_COMPLETE_METADATA
MISSING_LAYERS_CSV   = config.CSV_MISSING_LAYERS
HYDROBASINS_DIR      = config.HYDROBASINS_DIR

# Layer file → expected band count, derived from the ENABLED registry. Used only
# to REPORT missing layers, never to drop an activation. flood_mask is added
# because Script 4 produces it locally (it is not a GEE export).
def _expected_layers():
    layers = {spec.key: {"file": spec.filename, "bands": spec.band_count()}
              for spec in enabled_layers()}
    layers["flood_mask"] = {"file": "flood_mask.tif", "bands": 1}
    return layers

# Legacy GEE export name → canonical filename (kept for older exports on disk).
LAYER_RENAME = {
    "S1.tif":         "S1_VV_VH.tif",
    "S2_indices.tif": "S2_NDVI_NDBI.tif",
    "land_cover.tif": "S2_NDVI_NDBI.tif",
    "ESA_PW.tif":     "ESA_WorldCover_PermanentWater.tif",
}


# ─── EXPORT REORGANIZATION ───────────────────────────────────────────────────

def reorganize_export(export_root: Path) -> bool:
    """
    Bring a downloaded GEE export folder into the canonical flat structure:

      data/GEE_exports/{folder_name}/
        S1_VV_VH.tif        2 bands
        land_cover.tif      2 bands
        MERIT.tif           4 bands
        Soil.tif            2 bands
        ESA_WorldCover_PermanentWater.tif  1 band  permanent water (ESA WorldCover)
        Precipitation.tif  10 bands
        SoilMoisture.tif   10 bands

    Three things are handled:
      1. Flatten nested subfolder — GEE toDrive() with a slash in fileNamePrefix
         creates a subfolder inside the Drive folder; script 3 preserves that,
         so files land at export_root/{folder_name}/ instead of export_root/.
      2. Merge spatial tiles — when an export exceeds GEE's per-file limit it is
         split into tiles named {Layer}-XXXXXXXXXX-XXXXXXXXXX.tif.  Any number of
         tiles is supported; they are mosaicked and the originals deleted.
      3. Rename to canonical names (S1.tif → S1_VV_VH.tif, S2_indices.tif → land_cover.tif).

    Idempotent: already-reorganized folders are skipped at each sub-step.
    Returns True if the folder looks complete after reorganization.
    """
    # ── Step 1: flatten nested subfolder ─────────────────────────────────────
    for sub in list(export_root.iterdir()):
        if not sub.is_dir():
            continue
        tifs = list(sub.glob("*.tif"))
        if not tifs:
            continue
        for f in list(sub.iterdir()):
            dest = export_root / f.name
            if not dest.exists():
                f.rename(dest)
        # remove subfolder only if now empty
        remaining = list(sub.iterdir())
        if not remaining:
            sub.rmdir()

    # ── Step 2: merge spatial tiles → canonical single files ─────────────────
    # For each enabled layer, find its tiles "{stem}-XXXX-YYYY.tif" and mosaic
    # them. Temporal layers are date-stamped (Precipitation_YYYYMMDD_YYYYMMDD),
    # so their stem is discovered from disk and the merged file keeps that name.
    merge_specs = []  # (tile_prefix, out_name)
    for spec in enabled_layers():
        stem = spec.filename[:-4]
        if spec.kind == "temporal":
            # Discover the dated stem(s) present (one per temporal layer per event).
            dated = {t.name.split("-")[0] for t in export_root.glob(f"{stem}_*-*.tif")}
            dated |= {p.stem for p in export_root.glob(f"{stem}_*.tif") if "-" not in p.stem}
            for d in sorted(dated):
                merge_specs.append((d, f"{d}.tif"))
        else:
            merge_specs.append((stem, spec.filename))

    for prefix, out_name in merge_specs:
        out_path = export_root / out_name
        tiles    = sorted(export_root.glob(f"{prefix}-*.tif"))

        if not tiles:
            continue  # no tiles; single file or already merged

        if out_path.exists():
            # merged file already present — just clean up leftover tiles
            for t in tiles:
                t.unlink(missing_ok=True)
            continue

        # Validate tiles (skip 0-byte/corrupt files — they were incomplete downloads)
        valid_tiles = [t for t in tiles if t.stat().st_size > 0]
        if not valid_tiles:
            print(f"    ! all tiles for {prefix} are 0-byte — re-run script 3 to re-download")
            continue
        if len(valid_tiles) < len(tiles):
            bad = [t.name for t in tiles if t.stat().st_size == 0]
            print(f"    ! skipping corrupt (0-byte) tiles for {prefix}: {bad}")
            print(f"      Delete them and re-run script 3, then script 4 again")
            continue

        try:
            datasets = [rasterio.open(t) for t in valid_tiles]
            mosaic, transform = rio_merge(datasets)
            profile = datasets[0].profile.copy()
            profile.update(height=mosaic.shape[1], width=mosaic.shape[2],
                           transform=transform)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(mosaic)
            for ds in datasets:
                ds.close()
            for t in valid_tiles:
                t.unlink(missing_ok=True)
            print(f"    merged {len(valid_tiles)} {prefix} tiles → {out_path.name}")
        except Exception as e:
            print(f"    ! tile merge failed for {prefix}: {e}")

    # ── Step 3: rename to canonical filenames ─────────────────────────────────
    for old_name, new_name in LAYER_RENAME.items():
        src = export_root / old_name
        dst = export_root / new_name
        if src.exists():
            if not dst.exists():
                src.rename(dst)
            else:
                src.unlink()  # dst already produced by tile merge; drop the duplicate

    return True


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


def _resolve_layer_file(folder: Path, fname: str) -> Optional[Path]:
    """
    Locate a layer file in an event folder. Temporal layers are date-stamped
    (e.g. Precipitation_YYYYMMDD_YYYYMMDD.tif), so they are matched by glob on
    the filename stem; everything else is an exact name.
    """
    stem = fname[:-4]
    if stem in ("Precipitation", "SoilMoisture"):
        hits = sorted(folder.glob(f"{stem}_*.tif"))
        return hits[0] if hits else None
    p = folder / fname
    return p if p.exists() else None


def missing_layers_for(folder: Path) -> List[str]:
    """
    Return the list of ENABLED layers absent (or wrong band count) for one
    event folder. The caller decides gating: an absent CORE layer excludes the
    event from the catalog, an absent custom layer does not.
    """
    missing = []
    for key, spec in _expected_layers().items():
        path = _resolve_layer_file(folder, spec["file"])
        if path is None or not is_valid_tif(path, spec["bands"]):
            missing.append(key)
    return missing


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
    Get the HydroBASINS Pfafstetter Level-5 basin code for an activation, based on
    the AOI centroid. The Level-5 code is the first 5 digits of the PFAF_ID of the
    Level-12 basin the centroid falls in.

    Strategy:
    1. Try 'within' query first (centroid inside basin)
    2. If no match, find nearest basin (handles coastal/ocean areas)

    Args:
        aoi_shp_path: Path to AOI shapefile
        basins_gdf: GeoDataFrame of HydroBASINS level 12 (must include PFAF_ID)

    Returns:
        Level-5 Pfafstetter code as string (e.g. "23218"), or None if not found
    """
    def _l5(pfaf_id) -> str:
        return str(int(pfaf_id))[:5]

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

        if len(joined) > 0 and 'PFAF_ID' in joined.columns and not pd.isna(joined.iloc[0]['PFAF_ID']):
            return _l5(joined.iloc[0]['PFAF_ID'])

        # Try 2: Find nearest basin (for coastal/ocean areas)
        basins_gdf['distance'] = basins_gdf.geometry.distance(centroid)
        nearest_idx = basins_gdf['distance'].idxmin()

        # Only use nearest if it's reasonably close (within ~50km = ~0.5 degrees)
        if basins_gdf.loc[nearest_idx, 'distance'] < 0.5:
            return _l5(basins_gdf.loc[nearest_idx, 'PFAF_ID'])

        return None

    except Exception as e:
        return None


# ─── FLOOD MASK RASTERIZATION ────────────────────────────────────────────────

def rasterize_flood_mask(export_folder: Path, dcc_folder: Path) -> bool:
    """
    Rasterize flood_extent/event.shp to flood_mask.tif aligned to S1_VV_VH.tif.
    Binary output: 1 = flooded, 0 = not flooded. Skips if already exists.
    Assumes reorganize_export() has already been called on export_folder.
    Returns True on success or if already exists, False on failure.
    """
    out_tif   = export_folder / "flood_mask.tif"
    flood_shp = dcc_folder / "flood_extent" / "event.shp"
    s1_tif    = export_folder / "S1_VV_VH.tif"

    if out_tif.exists():
        return True

    if not flood_shp.exists():
        return False

    if not s1_tif.exists():
        return False

    try:
        with rasterio.open(s1_tif) as ref:
            crs       = ref.crs
            transform = ref.transform
            height    = ref.height
            width     = ref.width

        flood_gdf = gpd.read_file(flood_shp)
        if flood_gdf.crs and flood_gdf.crs.to_epsg() != crs.to_epsg():
            flood_gdf = flood_gdf.to_crs(crs)

        shapes = [(geom, 1) for geom in flood_gdf.geometry if geom is not None and not geom.is_empty]

        if shapes:
            mask = rio_rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype='uint8',
            )
        else:
            mask = np.zeros((height, width), dtype='uint8')

        profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'width': width,
            'height': height,
            'count': 1,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw',
        }
        with rasterio.open(out_tif, 'w', **profile) as dst:
            dst.write(mask, 1)

        return True

    except Exception as e:
        print(f"      ! flood_mask rasterization failed: {e}")
        return False


# ─── MAIN ────────────────────────────────────────────────────────────────────

def find_activation_in_exports(folder_name: str) -> Optional[Path]:
    """Return path to GEE_exports/{EMSR_code}/{folder_name}/ or None if not found."""
    emsr_code = folder_name.split("_")[0]
    p = GEE_EXPORTS_DIR / emsr_code / folder_name
    return p if p.exists() else None


def main():
    print("=" * 72)
    print("  GEE Output Preprocessing  (Script 4)")
    print(f"  BASE_DIR        : {BASE_DIR}")
    print(f"  GEE_EXPORTS_DIR : {GEE_EXPORTS_DIR}")
    print(f"  GEE_TASKS_CSV   : {GEE_TASKS_CSV}")
    print("=" * 72)

    config.migrate_csv_names()  # rename any old-named metadata files in place

    if not GEE_TASKS_CSV.exists():
        print(f"\n! {GEE_TASKS_CSV.name} not found: {GEE_TASKS_CSV}")
        print("  Run Script 2 first (--update-tracking) to generate it.")
        sys.exit(1)

    if not GEE_EXPORTS_DIR.exists():
        print(f"\n! GEE_EXPORTS_DIR not found: {GEE_EXPORTS_DIR}")
        print("  Run Script 3 first to download GEE exports.")
        sys.exit(1)

    META_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Reorganize all export folders into canonical flat structure ────
    print("\n[1/6] Reorganizing GEE export folders (flatten + merge tiles + rename)...")
    # Structure: GEE_exports/{EMSR_code}/{activation_folder}/
    all_export_dirs = sorted(
        act for emsr_dir in GEE_EXPORTS_DIR.iterdir() if emsr_dir.is_dir()
        for act in emsr_dir.iterdir() if act.is_dir()
    )
    for export_root in all_export_dirs:
        reorganize_export(export_root)
    print(f"  Done ({len(all_export_dirs)} folders)")

    # Load HydroBASINS
    print("\n[2/6] Loading HydroBASINS level-12 data...")
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
    print("\n[3/6] Loading gee_tasks_record.csv...")
    try:
        tracking_df = pd.read_csv(GEE_TASKS_CSV)
        print(f"  ✓ Loaded {len(tracking_df)} activations")
    except Exception as e:
        print(f"  ! Error: {e}")
        sys.exit(1)

    # Load activations.csv (sensor info from Script 1)
    print("\n[4/6] Loading activations.csv (sensor info)...")
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
            print(f"  ! Could not load activations.csv: {e} -- sensor columns will be empty")
    else:
        print(f"  ! activations.csv not found at {ACTIVATIONS_CSV} -- sensor columns will be empty")

    # Rasterize flood masks
    print("\n[5/6] Rasterizing flood masks from CEMS shapefiles...")
    mask_ok = mask_fail = mask_skip = 0
    for row in tracking_df.itertuples():
        folder_name   = row.folder_name
        emsr_code     = folder_name.split("_")[0]
        export_folder = find_activation_in_exports(folder_name)
        if export_folder is None:
            continue
        dcc_folder = DCC_ACTIVATIONS_DIR / emsr_code / folder_name
        if (export_folder / "flood_mask.tif").exists():
            mask_skip += 1
            continue
        ok = rasterize_flood_mask(export_folder, dcc_folder)
        if ok:
            mask_ok += 1
            print(f"  ✓ {folder_name}")
        else:
            mask_fail += 1
            print(f"  ✗ {folder_name} -- missing flood_extent/event.shp or S1_VV_VH.tif")
    print(f"  Done: {mask_ok} rasterized, {mask_skip} already existed, {mask_fail} failed")

    # Build the catalog. An event missing a CORE layer is incomplete: it is
    # written to the missing-layers report and left out of the catalog. Missing
    # only a custom (user-added) layer is tolerated, so the event stays in the
    # catalog and the absent custom layer is still recorded in the report.
    print("\n[6/6] Building catalog + missing-layers report...")
    catalog_records = []
    missing_records = []
    total = len(tracking_df)
    complete_count = 0
    gated_count = 0
    CORE_KEYS = set(core_keys())

    for idx, row in tracking_df.iterrows():
        folder_name = row['folder_name']
        emsr_code   = folder_name.split("_")[0]

        export_folder = find_activation_in_exports(folder_name)

        # Enabled layers absent for this event.
        if export_folder is None:
            missing = list(_expected_layers().keys())  # nothing downloaded yet
        else:
            missing = missing_layers_for(export_folder)
        core_missing = [k for k in missing if k in CORE_KEYS]

        if not missing:
            complete_count += 1
        else:
            missing_records.append({
                'folder_name': folder_name,
                'n_missing': len(missing),
                'missing_layers': ','.join(missing),
                'core_missing': ','.join(core_missing),
            })

        # Gate on core layers: an event missing any core layer is excluded from
        # the catalog (it lives in the missing-layers report instead).
        if core_missing:
            gated_count += 1
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total} events...")
            continue

        # Detect region from AOI shapefile
        aoi_shp = DCC_ACTIVATIONS_DIR / emsr_code / folder_name / "aoi" / "aoi.shp"
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

        catalog_records.append({
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

    fieldnames = [
        'folder_name', 'region', 'basin_id',
        'pre_event_sensor', 'post_event_sensors',
        'resolution_post_sensor', 'resolution_class',
    ]

    # The complete catalog is the accumulated record of every event ever
    # cataloged. Diff this run against it: an event whose folder_name is not yet
    # in the complete catalog is new and goes to 4_dataset_metadata.csv; the
    # complete catalog is then refreshed to old rows plus this run's events.
    DATASET_METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if COMPLETE_METADATA_CSV.exists():
        with open(COMPLETE_METADATA_CSV, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                existing[row['folder_name']] = row

    new_records = [r for r in catalog_records if r['folder_name'] not in existing]

    # New events from this run.
    with open(DATASET_METADATA_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_records)

    # Full accumulated catalog: prior rows are settled and never modified (this
    # preserves any 'split' Step 5 wrote). Only genuinely new events are added.
    merged = dict(existing)
    for r in new_records:
        merged[r['folder_name']] = r
    # Prior rows may carry extra columns (e.g. a 'split' added by Step 5); keep
    # them by extending the header with any field beyond the base set.
    extra = [k for k in next(iter(merged.values()), {}) if k not in fieldnames]
    complete_fields = fieldnames + extra
    with open(COMPLETE_METADATA_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=complete_fields, extrasaction='ignore')
        writer.writeheader()
        for row in merged.values():
            writer.writerow({k: row.get(k, '') for k in complete_fields})

    # Write the missing-layers report (only events with absences).
    with open(MISSING_LAYERS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['folder_name', 'n_missing', 'missing_layers', 'core_missing'])
        writer.writeheader()
        writer.writerows(missing_records)

    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Events cataloged this run    : {len(catalog_records)}")
    print(f"  New this run                 : {len(new_records)}")
    print(f"  Complete catalog total       : {len(merged)}")
    print(f"  Flood masks rasterized       : {mask_ok}  (skipped: {mask_skip}, failed: {mask_fail})")
    print(f"  Complete (all enabled layers): {complete_count}")
    print(f"  With missing layers          : {len(missing_records)}")
    print(f"  Excluded (missing core layer): {gated_count}")
    print()
    print(f"  new this run  -> {DATASET_METADATA_CSV}")
    print(f"  full catalog  -> {COMPLETE_METADATA_CSV}")
    print(f"  missing report-> {MISSING_LAYERS_CSV}")
    print("=" * 72)


if __name__ == "__main__":
    main()
