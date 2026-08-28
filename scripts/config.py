#!/usr/bin/env python3
"""
config.py — user-facing configuration for the flood-dataset pipeline.

This is the ONE file you edit to control what the pipeline produces:

  - LAYER_TOGGLES   turn each GEE layer on or off.
  - N_DAYS_OVERRIDE change the length of a temporal (daily) layer.

The layer definitions themselves live in add_gee_layers.py. config.py applies
your toggles / overrides to that registry and exposes `enabled_layers()`, which
Scripts 2 and 4 iterate over. Disabling a layer here means it is never exported
and never reported as "missing" — the pipeline produces a clean, smaller dataset
with no failures.

Also holds the shared metadata paths and the step-numbered CSV names so every
script agrees on where things live.
"""

from pathlib import Path
from typing import Dict, List

import add_gee_layers
from add_gee_layers import LayerSpec


# ─── 1. LAYER TOGGLES ─────────────────────────────────────────────────────────
# Set a layer to False to skip it entirely. Custom layers added in
# add_gee_layers.py default to enabled unless you list them here as False.

LAYER_TOGGLES: Dict[str, bool] = {
    "S1":            True,
    "S2_indices":    True,
    "MERIT":         True,
    "Soil":          True,
    "ESA_PW":        True,
    "Precipitation": True,
    "SoilMoisture":  True,
}


# ─── 2. TEMPORAL WINDOW LENGTH ────────────────────────────────────────────────
# Per-temporal-layer number of antecedent days (event day excluded). Overrides
# the n_days baked into the registry. Omit a layer to keep its registry default.
# The exported band count follows automatically, so changing this never causes a
# validation failure downstream.

N_DAYS_OVERRIDE: Dict[str, int] = {
    "Precipitation": 30,
    "SoilMoisture":  30,
}


# ─── 3. TEMPORAL-LAYER AOI BUFFER ─────────────────────────────────────────────
# How far the AOI is expanded, in degrees, before the coarse temporal layers
# (precipitation, soil moisture) are exported. Each location later keeps this
# neighbourhood of weather pixels around it instead of a single resampled value.
# Increase it to capture upstream rainfall and soil moisture that drive
# downstream discharge; 0.6 degree is ~67 km at the equator.

TEMPORAL_BUFFER_DEG: float = 0.6

# How Script 2 delivers the layers:
#   "direct"  download straight from GEE onto disk (tiled getDownloadURL into
#             data/GEE_exports/). No Drive account needed, no Script 3 — go
#             straight to Script 4 preprocessing. Runs synchronously, so a big
#             batch keeps the machine busy while it downloads.
#   "drive"   the original path: submit Export.image.toDrive tasks, then fetch
#             the results with Script 3. Better for very large batches because
#             GEE renders exports server-side while nothing runs locally.
EXPORT_MODE: str = "direct"


# ─── 4. PATCH TILING (Step 6) ─────────────────────────────────────────────────
# Step 6 cuts each event's co-registered layers into square, non-overlapping
# patches. PATCH_SIZE_M is the ground side of a patch; STRIDE_M is the step
# between patch origins (= PATCH_SIZE_M means no overlap). MIN_VALID_RATIO drops
# a patch whose 10 m stack is mostly nodata. The per-resolution pixel grids
# (10 m / 80 m / 160 m / 2560 m) follow from PATCH_SIZE_M and the source layer
# resolutions, so a 2560 m patch is 256x256 at 10 m, 32x32 at 80 m, etc.

PATCH_SIZE_M: int   = 2560    # 2.56 km square patches
STRIDE_M: int       = 2560    # no overlap
MIN_VALID_RATIO: float = 0.1  # drop a patch with <10% valid 10 m pixels
PATCH_NODATA: int   = 9999    # nodata for input stacks (flood mask uses 0)


# ─── PATHS ────────────────────────────────────────────────────────────────────

BASE_DIR            = Path(__file__).resolve().parent.parent
DATA_DIR            = BASE_DIR / "data"
META_DIR            = DATA_DIR / "metadata"
GEE_EXPORTS_DIR     = DATA_DIR / "GEE_exports"
ACTIVATIONS_RAW_DIR = DATA_DIR / "activations" / "activations_raw"        # Step 1 raw downloads
ACTIVATIONS_DIR     = DATA_DIR / "activations" / "activations_reorganized"
TEMP_DIR            = DATA_DIR / "_temp"        # Step 1 scratch space for product ZIPs
HYDROBASINS_DIR     = DATA_DIR / "hydrobasins"
CONTINENTS_DIR      = DATA_DIR / "continents"   # Natural Earth countries (continent field)
CLIMATE_DIR         = DATA_DIR / "climate"      # Beck Koppen-Geiger raster

# Step 3 (drive mode only) — Google Drive OAuth
GDRIVE_TOKEN_FILE   = DATA_DIR / ".gdrive_token.json"
GDRIVE_CREDS_DIR    = BASE_DIR / "Gdrive_credentials"


# ─── STEP-NUMBERED METADATA FILES ─────────────────────────────────────────────
# Prefix = the pipeline step that writes the file.

# Step 1 — EMSR acquisition + standardization
CSV_ACTIVATION_STATUS   = META_DIR / "1_activation_status.csv"   # per-product resume state
CSV_ACTIVATION_SENSORS  = META_DIR / "1_activation_sensors.csv"  # pre/post sensor + date from PDF
CSV_ACTIVATION_CATALOG  = META_DIR / "1_activation_catalog.csv"  # first-draft catalog

# Step 2 — GEE export submission
CSV_GEE_EXPORT_STATUS   = META_DIR / "2_gee_export_status.csv"   # per-layer export status
# Provenance of the S1/S2 median composites: which acquisition window, cloud
# threshold and image count each exported composite was actually built from.
# Not needed by any later step — kept so a released composite can be traced
# back to its source imagery.
CSV_COMPOSITE_REGISTRY  = META_DIR / "2_composite_registry.csv"

# Step 4 — metadata compilation
CSV_DATASET_METADATA    = META_DIR / "4_dataset_metadata.csv"    # events new in the latest run
CSV_COMPLETE_METADATA   = META_DIR / "released_events_metadata.csv"  # full accumulated catalog (one row per event)
CSV_MISSING_LAYERS      = META_DIR / "4_missing_layers_report.csv"  # missing enabled layers/event

# Step 5 — patch tiling
PATCHES_DIR             = DATA_DIR / "patches"                       # patch tiles
CSV_PATCH_METADATA      = META_DIR / "released_patches_metadata.csv"  # one row per patch
CSV_PATCH_VALIDATION    = META_DIR / "5_patch_validation_issues.csv"  # QC findings

# Step 6 — split generation (runs after patching, balances by patch count)
SPLIT_DIR               = META_DIR / "split_global"                  # the three split index files
CSV_TRAIN_PATCHES       = SPLIT_DIR / "train_patches.csv"
CSV_VAL_PATCHES         = SPLIT_DIR / "val_patches.csv"
CSV_TEST_PATCHES        = SPLIT_DIR / "test_patches.csv"
PLOTS_DIR               = DATA_DIR / "plots" / "splits"             # split balance plots

# ─── LOCAL PATH OVERRIDES (optional) ──────────────────────────────────────────
# To run the pipeline against a data tree outside the repo, create
# scripts/config_local.py and redefine any of the paths above in it. Setting
# DATA_DIR alone is enough — every directory and CSV below it is rebuilt from
# the new root afterwards:
#
#     from pathlib import Path
#     DATA_DIR = Path("/data/flood")
#
# To move one thing on its own, name it explicitly and it is kept as given:
#
#     from pathlib import Path
#     DATA_DIR        = Path("/data/flood")
#     GEE_EXPORTS_DIR = Path("/mnt/bigdisk/GEE_exports")   # elsewhere
#
# The file is gitignored, so machine-specific paths stay out of the repo.
_DEFAULTS = dict(vars())          # snapshot of the repo defaults, before override

# Which names config_local actually assigns, read from the module itself rather
# than inferred by comparing values — a user who deliberately pins a path to the
# same value as the default must still have that treated as an explicit choice.
_LOCAL_NAMES: set = set()
try:
    import config_local as _cl
    _LOCAL_NAMES = {n for n in vars(_cl) if not n.startswith("_")}
    from config_local import *  # noqa: F401,F403
    _HAS_LOCAL = True
except ImportError:
    _HAS_LOCAL = False


def _rebase_derived_paths() -> None:
    """
    Re-derive every path under DATA_DIR after a config_local override.

    Paths here are plain module constants, computed at import time, so a
    config_local that sets only DATA_DIR would otherwise leave all of them
    pointing at the old root — the whole pipeline would keep writing to
    data/ while the user believed it had moved. Anything config_local names
    explicitly is left exactly as given.
    """
    g = globals()
    if not _HAS_LOCAL or "DATA_DIR" not in _LOCAL_NAMES:
        return

    base = g["DATA_DIR"]
    meta = g["META_DIR"] if "META_DIR" in _LOCAL_NAMES else base / "metadata"

    derived = {
        "META_DIR":            meta,
        "GEE_EXPORTS_DIR":     base / "GEE_exports",
        "ACTIVATIONS_RAW_DIR": base / "activations" / "activations_raw",
        "ACTIVATIONS_DIR":     base / "activations" / "activations_reorganized",
        "TEMP_DIR":            base / "_temp",
        "HYDROBASINS_DIR":     base / "hydrobasins",
        "CONTINENTS_DIR":      base / "continents",
        "CLIMATE_DIR":         base / "climate",
        "GDRIVE_TOKEN_FILE":   base / ".gdrive_token.json",
        "PATCHES_DIR":         base / "patches",
        "PLOTS_DIR":           base / "plots" / "splits",
        "SPLIT_DIR":           meta / "split_global",
        "CSV_ACTIVATION_STATUS":  meta / "1_activation_status.csv",
        "CSV_ACTIVATION_SENSORS": meta / "1_activation_sensors.csv",
        "CSV_ACTIVATION_CATALOG": meta / "1_activation_catalog.csv",
        "CSV_GEE_EXPORT_STATUS":  meta / "2_gee_export_status.csv",
        "CSV_COMPOSITE_REGISTRY": meta / "2_composite_registry.csv",
        "CSV_DATASET_METADATA":   meta / "4_dataset_metadata.csv",
        "CSV_COMPLETE_METADATA":  meta / "released_events_metadata.csv",
        "CSV_MISSING_LAYERS":     meta / "4_missing_layers_report.csv",
        "CSV_PATCH_METADATA":     meta / "released_patches_metadata.csv",
        "CSV_PATCH_VALIDATION":   meta / "5_patch_validation_issues.csv",
    }
    for name, value in derived.items():
        # Only rebase what config_local did not set itself.
        if name not in _LOCAL_NAMES:
            g[name] = value

    # The three split CSVs live under SPLIT_DIR, which may itself have moved.
    for name, fname in (("CSV_TRAIN_PATCHES", "train_patches.csv"),
                        ("CSV_VAL_PATCHES",   "val_patches.csv"),
                        ("CSV_TEST_PATCHES",  "test_patches.csv")):
        if name not in _LOCAL_NAMES:
            g[name] = g["SPLIT_DIR"] / fname


_rebase_derived_paths()


# Map old filenames -> new, for one-time on-disk migration (see migrate_csv_names).
CSV_MIGRATION = {
    "activations_status.csv":   CSV_ACTIVATION_STATUS.name,
    "activations_sources.csv":  CSV_ACTIVATION_SENSORS.name,
    "activations.csv":          CSV_ACTIVATION_CATALOG.name,
    "gee_tasks_record.csv":     CSV_GEE_EXPORT_STATUS.name,
    "dataset_metadata.csv":     CSV_DATASET_METADATA.name,
    "complete_dataset_metadata.csv": CSV_COMPLETE_METADATA.name,
    "patch_metadata.csv":       CSV_PATCH_METADATA.name,
}


def migrate_csv_names() -> None:
    """
    Rename any pre-existing metadata files from their old names to the new
    step-numbered names, preserving run state. Safe to call on every run:
    already-migrated or absent files are skipped, and an existing new file is
    never overwritten.
    """
    for old, new in CSV_MIGRATION.items():
        src = META_DIR / old
        dst = META_DIR / new
        if src.exists() and not dst.exists():
            src.rename(dst)
            print(f"  migrated {old} -> {new}")


# ─── ENABLED-LAYER VIEW OF THE REGISTRY ───────────────────────────────────────

def _apply_overrides(spec: LayerSpec) -> LayerSpec:
    """Return spec with N_DAYS_OVERRIDE applied (temporal layers only)."""
    if spec.kind == "temporal" and spec.key in N_DAYS_OVERRIDE:
        spec.n_days = int(N_DAYS_OVERRIDE[spec.key])
    return spec


def is_enabled(key: str) -> bool:
    """A layer is enabled unless explicitly toggled off."""
    return LAYER_TOGGLES.get(key, True)


def enabled_layers() -> List[LayerSpec]:
    """
    The layers the pipeline should produce this run: registry order, custom
    layers appended, filtered by LAYER_TOGGLES, with n_days overrides applied.
    Scripts 2 and 4 both iterate this list — the single source of truth for a run.
    """
    out = []
    for spec in add_gee_layers.all_layers():
        if is_enabled(spec.key):
            out.append(_apply_overrides(spec))
    return out


def enabled_keys() -> List[str]:
    return [s.key for s in enabled_layers()]


if __name__ == "__main__":
    print("Enabled layers this run:")
    for s in enabled_layers():
        nd = f", n_days={s.n_days}" if s.kind == "temporal" else ""
        print(f"  {s.key:14s} -> {s.filename:20s} "
              f"[{s.kind}, {s.band_count()} bands{nd}]")
