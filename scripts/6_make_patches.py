#!/usr/bin/env python3
"""
Script 6: Make patches

Cuts each cataloged event's co-registered GeoTIFFs into square, non-overlapping
patches and writes them as individual GeoTIFFs, then validates every patch.

A patch covers PATCH_SIZE_M x PATCH_SIZE_M on the ground (2.56 km by default)
and is written at four resolutions plus the label, five files in all:

  patch_NNNN_input_10m.tif    256x256, 5 bands   S1 VV, S1 VH, NDVI, NDBI, permanent water
  patch_NNNN_input_80m.tif     32x32 , 5 bands   MERIT elevation, flowdir sin, flowdir cos, UDA, HAND
  patch_NNNN_input_160m.tif    16x16 , 2 bands   SoilGrids clay, sand
  patch_NNNN_input_2560m.tif    1x1  , 2N bands  Precipitation (N) + SoilMoisture (N), N=30 default
  patch_NNNN_flood_mask.tif   256x256, 1 band    CEMS flood extent (1 = flooded)

The flood mask is the CEMS delineation only (flood_mask.tif from Step 4). It is
never altered with permanent water. Permanent water is provided as a separate
input band (band 5 of input_10m), so a model can distinguish pre-existing water
from new flooding but the label stays the raw observed inundation.

MERIT flow direction (D8) is encoded as (sin, cos) of its compass angle so the
circular variable has no artificial discontinuity at 0/360 degrees.

Input
  data/GEE_exports/{EMSR}/{folder}/  S1_VV_VH, S2_NDVI_NDBI, MERIT, Soil,
                                     ESA_WorldCover_PermanentWater, Precipitation,
                                     SoilMoisture, flood_mask  (all from Steps 2-4)
  data/metadata/complete_dataset_metadata.csv   the catalog (one row per event)

Output
  data/patches/{EMSR}/{folder}/patch_NNNN_*.tif
  data/metadata/patch_metadata.csv              one row per patch (+ split, basin)
  data/metadata/6_patch_validation_issues.csv   QC findings, if any

Usage
  python scripts/6_make_patches.py
"""

import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds, Affine
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.warp import reproject, Resampling
except ImportError:
    print("ERROR: rasterio not found. Install with: pip install rasterio")
    sys.exit(1)

try:
    import geopandas as gpd
except ImportError:
    print("ERROR: geopandas not found. Install with: pip install geopandas")
    sys.exit(1)

import config

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR            = config.BASE_DIR
GEE_EXPORTS_DIR     = config.GEE_EXPORTS_DIR
ACTIVATIONS_DIR     = config.ACTIVATIONS_DIR
PATCHES_DIR         = config.PATCHES_DIR
META_DIR            = config.META_DIR
CATALOG_CSV         = config.CSV_COMPLETE_METADATA
PATCH_METADATA_CSV  = config.CSV_PATCH_METADATA
VALIDATION_CSV      = config.CSV_PATCH_VALIDATION

# ── PATCH GEOMETRY (from config) ──────────────────────────────────────────────
PATCH_SIZE_M    = config.PATCH_SIZE_M
STRIDE_M        = config.STRIDE_M
MIN_VALID_RATIO = config.MIN_VALID_RATIO
NODATA          = config.PATCH_NODATA

# Resolution (m) of each stack and the per-patch pixel grid it produces.
RES_10M, RES_80M, RES_160M, RES_2560M = 10.0, 80.0, 160.0, 2560.0
PX_10M   = int(PATCH_SIZE_M / RES_10M)     # 256
PX_80M   = int(PATCH_SIZE_M / RES_80M)     # 32
PX_160M  = int(PATCH_SIZE_M / RES_160M)    # 16
PX_2560M = int(PATCH_SIZE_M / RES_2560M)   # 1

# Canonical per-event layer filenames (post Step 4 rename).
F_S1    = "S1_VV_VH.tif"
F_S2    = "S2_NDVI_NDBI.tif"
F_MERIT = "MERIT.tif"
F_SOIL  = "Soil.tif"
F_PW    = "ESA_WorldCover_PermanentWater.tif"
F_MASK  = "flood_mask.tif"
# Temporal layers are date-stamped per event (Precipitation_YYYYMMDD_YYYYMMDD.tif),
# so they are located by glob, not a fixed name.
G_PRE   = "Precipitation_*.tif"
G_SM    = "SoilMoisture_*.tif"

# Temporal band count per layer (event day excluded). Drives input_2560m width.
N_DAYS  = config.N_DAYS_OVERRIDE.get("Precipitation", 30)
N_2560M = 2 * N_DAYS   # precipitation days + soil-moisture days

# D8 flow direction code -> compass angle (degrees clockwise from East).
D8_ANGLE = {1: 0, 2: 45, 4: 90, 8: 135, 16: 180, 32: 225, 64: 270, 128: 315}

# Per-patch file -> (expected bands, expected H, W) for validation.
EXPECTED = {
    "input_10m":   (5, PX_10M,   PX_10M),
    "input_80m":   (5, PX_80M,   PX_80M),
    "input_160m":  (2, PX_160M,  PX_160M),
    "input_2560m": (N_2560M, PX_2560M, PX_2560M),
    "flood_mask":  (1, PX_10M,   PX_10M),
}


def _find_one(folder: Path, pattern: str) -> Optional[Path]:
    """Return the single file matching a glob pattern in folder, or None."""
    hits = sorted(folder.glob(pattern))
    return hits[0] if hits else None


# ─── STACK BUILDERS ───────────────────────────────────────────────────────────

def _reproject_band(src, band_idx, dst, transform, crs, resampling):
    """Reproject one source band into dst (in place), normalising nodata to NODATA."""
    arr = src.read(band_idx).astype(np.float32)
    arr[~np.isfinite(arr)] = NODATA
    if src.nodata is not None:
        arr[arr == src.nodata] = NODATA
    reproject(
        source=arr, destination=dst,
        src_transform=src.transform, src_crs=src.crs,
        dst_transform=transform, dst_crs=crs,
        resampling=resampling, src_nodata=NODATA, dst_nodata=NODATA,
    )


def _grid(ref_bounds, res) -> Tuple[int, int, Affine]:
    minx, miny, maxx, maxy = ref_bounds
    width  = int((maxx - minx) / res)
    height = int((maxy - miny) / res)
    return width, height, from_bounds(minx, miny, maxx, maxy, width, height)


def build_stack_10m(gee: Path, ref_bounds, ref_crs):
    """S1 VV, S1 VH, NDVI, NDBI, permanent_water -> (5, H, W) at 10 m."""
    w, h, transform = _grid(ref_bounds, RES_10M)
    stack = np.full((5, h, w), NODATA, dtype=np.float32)
    s1 = gee / F_S1
    if s1.exists():
        with rasterio.open(s1) as src:
            for i, b in enumerate((1, 2)):
                _reproject_band(src, b, stack[i], transform, ref_crs, Resampling.cubic)
    s2 = gee / F_S2
    if s2.exists():
        with rasterio.open(s2) as src:
            for i, b in enumerate((1, 2)):
                _reproject_band(src, b, stack[2 + i], transform, ref_crs, Resampling.cubic)
    pw = gee / F_PW
    if pw.exists():
        with rasterio.open(pw) as src:
            # permanent water is a 0/1 mask: nearest, keep binary, no nodata fill
            arr = src.read(1).astype(np.float32)
            dst = np.zeros((h, w), dtype=np.float32)
            reproject(
                source=arr, destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=ref_crs,
                resampling=Resampling.nearest,
            )
            stack[4] = (dst > 0).astype(np.float32)
    else:
        stack[4] = 0.0
    return stack, transform


def build_stack_80m(gee: Path, ref_bounds, ref_crs):
    """MERIT elevation, flowdir sin, flowdir cos, UDA, HAND -> (5, H, W) at 80 m."""
    w, h, transform = _grid(ref_bounds, RES_80M)
    stack = np.full((5, h, w), NODATA, dtype=np.float32)
    merit = gee / F_MERIT
    if merit.exists():
        with rasterio.open(merit) as src:
            _reproject_band(src, 1, stack[0], transform, ref_crs, Resampling.cubic)  # elev
            # flow direction -> sin/cos (nearest; discrete codes)
            d8 = src.read(2).astype(np.float32)
            sin_a = np.full(d8.shape, NODATA, dtype=np.float32)
            cos_a = np.full(d8.shape, NODATA, dtype=np.float32)
            for code, ang in D8_ANGLE.items():
                m = d8 == code
                sin_a[m] = math.sin(math.radians(ang))
                cos_a[m] = math.cos(math.radians(ang))
            for arr, slot in ((sin_a, 1), (cos_a, 2)):
                reproject(
                    source=arr, destination=stack[slot],
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=ref_crs,
                    resampling=Resampling.nearest, src_nodata=NODATA, dst_nodata=NODATA,
                )
            _reproject_band(src, 3, stack[3], transform, ref_crs, Resampling.cubic)  # UDA
            _reproject_band(src, 4, stack[4], transform, ref_crs, Resampling.cubic)  # HAND
    return stack, transform


def build_stack_160m(gee: Path, ref_bounds, ref_crs):
    """SoilGrids clay, sand -> (2, H, W) at 160 m."""
    w, h, transform = _grid(ref_bounds, RES_160M)
    stack = np.full((2, h, w), NODATA, dtype=np.float32)
    soil = gee / F_SOIL
    if soil.exists():
        with rasterio.open(soil) as src:
            for i, b in enumerate((1, 2)):
                _reproject_band(src, b, stack[i], transform, ref_crs, Resampling.cubic)
    return stack, transform


def build_stack_2560m(gee: Path, ref_bounds, ref_crs):
    """
    Precipitation (N_DAYS) + SoilMoisture (N_DAYS) -> (2*N_DAYS, H, W) at 2560 m.
    Reads every band of each dated temporal file (no cap), so the patch follows
    the configured antecedent-window length.
    """
    w, h, transform = _grid(ref_bounds, RES_2560M)
    stack = np.full((N_2560M, h, w), NODATA, dtype=np.float32)
    for pattern, base in ((G_PRE, 0), (G_SM, N_DAYS)):
        p = _find_one(gee, pattern)
        if p is None:
            continue
        with rasterio.open(p) as src:
            for i in range(min(src.count, N_DAYS)):
                _reproject_band(src, i + 1, stack[base + i], transform, ref_crs, Resampling.average)
    return stack, transform


def build_flood_mask(gee: Path, ref_bounds, ref_crs):
    """
    Flood mask at 10 m, CEMS delineation only. Prefer the event's flood_mask.tif
    (produced by Step 4); fall back to rasterising flood_extent/event.shp.
    Never merges permanent water.
    """
    w, h, transform = _grid(ref_bounds, RES_10M)
    mask_tif = gee / F_MASK
    if mask_tif.exists():
        out = np.zeros((h, w), dtype=np.float32)
        with rasterio.open(mask_tif) as src:
            arr = src.read(1).astype(np.float32)
            reproject(
                source=arr, destination=out,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=ref_crs,
                resampling=Resampling.nearest,
            )
        return (out > 0).astype(np.uint8), transform
    return np.zeros((h, w), dtype=np.uint8), transform


# ─── PATCH EXTRACTION ─────────────────────────────────────────────────────────

def patch_grid(width, height) -> List[Tuple[int, int]]:
    """(row, col) origins in 10 m pixel space for a full tiling of the event."""
    size = PX_10M
    step = int(STRIDE_M / RES_10M)
    out = []
    r = 0
    while r + size <= height:
        c = 0
        while c + size <= width:
            out.append((r, c))
            c += step
        r += step
    return out


def has_valid_data(patch) -> bool:
    return (np.sum(patch != NODATA) / patch.size) >= MIN_VALID_RATIO


def write_patch(stacks, transforms, ref_crs, idx, r10, c10, out_dir) -> Optional[Dict]:
    """
    Extract + write one patch across all resolutions. Returns a metadata dict on
    success, or None if the patch is mostly nodata (skipped).
    """
    s10, s80, s160, s2560, mask = stacks
    t10, t80, t160, t2560, tmask = transforms
    name = f"patch_{idx:04d}"

    p10 = s10[:, r10:r10 + PX_10M, c10:c10 + PX_10M]
    if not has_valid_data(p10):
        return None

    r80, c80     = r10 // 8,   c10 // 8
    r160, c160   = r10 // 16,  c10 // 16
    r2560, c2560 = r10 // 256, c10 // 256

    p80   = s80[:, r80:r80 + PX_80M, c80:c80 + PX_80M]
    p160  = s160[:, r160:r160 + PX_160M, c160:c160 + PX_160M]
    p2560 = s2560[:, r2560:r2560 + PX_2560M, c2560:c2560 + PX_2560M]
    pmask = mask[r10:r10 + PX_10M, c10:c10 + PX_10M][np.newaxis]

    out_dir.mkdir(parents=True, exist_ok=True)
    items = [
        (p10,   f"{name}_input_10m.tif",   t10,   c10,   r10,   "float32"),
        (p80,   f"{name}_input_80m.tif",   t80,   c80,   r80,   "float32"),
        (p160,  f"{name}_input_160m.tif",  t160,  c160,  r160,  "float32"),
        (p2560, f"{name}_input_2560m.tif", t2560, c2560, r2560, "float32"),
        (pmask, f"{name}_flood_mask.tif",  tmask, c10,   r10,   "uint8"),
    ]
    for data, fname, base_t, coff, roff, dtype in items:
        t = base_t * Affine.translation(coff, roff)
        profile = {
            "driver": "GTiff", "height": data.shape[1], "width": data.shape[2],
            "count": data.shape[0], "dtype": dtype, "crs": ref_crs,
            "transform": t, "compress": "lzw",
            "nodata": None if dtype == "uint8" else NODATA,
        }
        with rasterio.open(out_dir / fname, "w", **profile) as dst:
            dst.write(data.astype(dtype))

    # Patch bounds from its 10 m transform.
    t = t10 * Affine.translation(c10, r10)
    minx, maxy = t * (0, 0)
    maxx, miny = t * (PX_10M, PX_10M)
    return {
        "patch_number": idx,
        "crs": str(ref_crs),
        "bounds_minx": minx, "bounds_miny": miny,
        "bounds_maxx": maxx, "bounds_maxy": maxy,
        "flood_pixels": int((pmask == 1).sum()),
    }


# ─── VALIDATION ───────────────────────────────────────────────────────────────

def validate_patch(out_dir: Path, idx: int) -> List[Dict]:
    """Check one patch's five files; fix NaN/Inf in place. Return issue dicts."""
    issues = []
    name = f"patch_{idx:04d}"
    for key, (bands, H, W) in EXPECTED.items():
        fpath = out_dir / f"{name}_{key}.tif"
        is_mask = key == "flood_mask"
        if not fpath.exists():
            issues.append({"patch": idx, "file": key, "check": "missing", "detail": "absent"})
            continue
        with rasterio.open(fpath) as src:
            data = src.read().astype(np.float32)
            profile = src.profile.copy()
        if data.shape[0] != bands:
            issues.append({"patch": idx, "file": key, "check": "band_count",
                           "detail": f"expected {bands}, got {data.shape[0]}"})
        if data.shape[1:] != (H, W):
            issues.append({"patch": idx, "file": key, "check": "shape",
                           "detail": f"expected {(H, W)}, got {tuple(data.shape[1:])}"})
        bad = ~np.isfinite(data)
        if bad.any():
            issues.append({"patch": idx, "file": key, "check": "nan_inf",
                           "detail": f"{int(bad.sum())} non-finite -> {'0' if is_mask else NODATA}"})
            data[bad] = 0.0 if is_mask else NODATA
            with rasterio.open(fpath, "w", **profile) as dst:
                dst.write(data.astype("uint8" if is_mask else "float32"))
        if is_mask:
            uniq = set(np.unique(data).tolist())
            if not uniq <= {0.0, 1.0}:
                issues.append({"patch": idx, "file": key, "check": "mask_not_binary",
                               "detail": f"values {sorted(uniq)[:5]}"})
    return issues


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def find_gee_folder(folder_name: str) -> Optional[Path]:
    p = GEE_EXPORTS_DIR / folder_name.split("_")[0] / folder_name
    return p if p.exists() else None


def main():
    print("=" * 80)
    print("  Script 6: Make patches")
    print(f"  Patch size : {PATCH_SIZE_M} m  ({PX_10M}x{PX_10M} @ 10 m), stride {STRIDE_M} m")
    print(f"  Output     : {PATCHES_DIR}")
    print("=" * 80)

    config.migrate_csv_names()

    if not CATALOG_CSV.exists():
        print(f"\nERROR: catalog not found: {CATALOG_CSV}\n  Run Steps 4 and 5 first.")
        sys.exit(1)

    with open(CATALOG_CSV, newline="", encoding="utf-8") as f:
        catalog = list(csv.DictReader(f))
    print(f"\nCataloged events: {len(catalog)}")

    META_DIR.mkdir(parents=True, exist_ok=True)
    patch_rows: List[Dict] = []
    all_issues: List[Dict] = []
    n_events = n_patches = 0

    for i, row in enumerate(catalog, 1):
        folder_name = row["folder_name"]
        emsr = folder_name.split("_")[0]
        gee = find_gee_folder(folder_name)
        if gee is None:
            print(f"[{i}/{len(catalog)}] {folder_name}  -- no GEE export, skipped")
            continue

        out_dir = PATCHES_DIR / emsr / folder_name
        if out_dir.exists() and list(out_dir.glob("patch_0000_*")):
            print(f"[{i}/{len(catalog)}] {folder_name}  -- already done, skipped")
            continue

        merit = gee / F_MERIT
        if not merit.exists():
            print(f"[{i}/{len(catalog)}] {folder_name}  -- no MERIT.tif, skipped")
            continue
        with rasterio.open(merit) as src:
            ref_crs, ref_bounds = src.crs, src.bounds

        t0 = time.time()
        s10,   t10   = build_stack_10m(gee, ref_bounds, ref_crs)
        s80,   t80   = build_stack_80m(gee, ref_bounds, ref_crs)
        s160,  t160  = build_stack_160m(gee, ref_bounds, ref_crs)
        s2560, t2560 = build_stack_2560m(gee, ref_bounds, ref_crs)
        mask,  tmask = build_flood_mask(gee, ref_bounds, ref_crs)
        stacks     = (s10, s80, s160, s2560, mask)
        transforms = (t10, t80, t160, t2560, tmask)

        grid = patch_grid(s10.shape[2], s10.shape[1])
        saved = 0
        for idx, (r10, c10) in enumerate(grid):
            meta = write_patch(stacks, transforms, ref_crs, idx, r10, c10, out_dir)
            if meta is None:
                continue
            meta.update({
                "emsr_code": emsr, "folder_name": folder_name,
                "split": row.get("split", ""),
                "basin_id": row.get("basin_id", ""),
                "resolution_class": row.get("resolution_class", ""),
                "resolution_post_sensor_m": row.get("resolution_post_sensor", ""),
            })
            for iss in validate_patch(out_dir, idx):
                iss.update({"emsr_code": emsr, "folder_name": folder_name})
                all_issues.append(iss)
            patch_rows.append(meta)
            saved += 1

        n_events += 1
        n_patches += saved
        print(f"[{i}/{len(catalog)}] {folder_name}  -- {saved}/{len(grid)} patches "
              f"[{time.time() - t0:.0f}s]")

    # ── write patch metadata ──────────────────────────────────────────────────
    if patch_rows:
        for j, r in enumerate(patch_rows):
            r["patch_index"] = j
        fields = ["patch_index", "emsr_code", "folder_name", "patch_number",
                  "crs", "bounds_minx", "bounds_miny", "bounds_maxx", "bounds_maxy",
                  "flood_pixels", "basin_id", "split",
                  "resolution_post_sensor_m", "resolution_class"]
        with open(PATCH_METADATA_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(patch_rows)
        print(f"\n  Wrote {PATCH_METADATA_CSV} ({len(patch_rows)} patches)")

    if all_issues:
        with open(VALIDATION_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["emsr_code", "folder_name", "patch", "file", "check", "detail"],
                extrasaction="ignore")
            w.writeheader()
            w.writerows(all_issues)
        print(f"  Wrote {VALIDATION_CSV} ({len(all_issues)} issues)")

    print("\n" + "=" * 80)
    print(f"DONE  events={n_events}  patches={n_patches}  "
          f"validation_issues={len(all_issues)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
