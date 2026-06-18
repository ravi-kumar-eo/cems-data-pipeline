# CEMS Multi-Resolution Flood Dataset

A ready-to-train, multi-modal global flood dataset built from Copernicus EMSR rapid-mapping activations and Earth-observation layers. The dataset ships as model-ready **patch tiles** with an activation- and basin-exclusive train/validation/test split, so you can download it and start training without touching Google Earth Engine. The open pipeline that builds it is included, so the dataset can also be reproduced or extended to new flood events.

**Dataset:** [Zenodo DOI — to be added]

## Two ways to use this

- **Train now — download the patches.** Get the patch dataset from Zenodo and load the tiles directly. Each 2.56 km tile carries co-registered inputs at four resolutions plus the flood label, with the train/val/test split already assigned. No GEE account, no building required.
- **Build or extend — run the pipeline.** Use the scripts to regenerate the dataset or extend it to flood activations newer than the release. This needs a GEE account and is described under *Building or extending the dataset* below.

## The patch dataset

Each cataloged flood event is cut into square, non-overlapping **2.56 km** tiles. Every tile is five co-registered GeoTIFFs — four input groups at their native resolutions plus the flood label:

| File | Bands | Pixels | Contents |
|---|---|---|---|
| `patch_NNNN_input_10m.tif` | 5 | 256×256 | S1 VV, S1 VH, NDVI, NDBI, permanent water |
| `patch_NNNN_input_80m.tif` | 5 | 32×32 | MERIT elevation, flow-dir sin, flow-dir cos, UDA, HAND |
| `patch_NNNN_input_160m.tif` | 2 | 16×16 | SoilGrids clay %, sand % |
| `patch_NNNN_input_2560m.tif` | 2N | 1×1 | precipitation (N days) + soil moisture (N days) |
| `patch_NNNN_flood_mask.tif` | 1 | 256×256 | flood label (1 = flooded), CEMS delineation only |

The flood mask is the official CEMS delineation alone. Permanent water (ESA WorldCover) is provided as a separate input band (band 5 of `input_10m`), so a model can tell pre-existing water from new flooding while the label stays the observed inundation. MERIT flow direction is encoded as the sine and cosine of its compass angle. `N` is the antecedent-window length (30 days by default), so `input_2560m` is 60 bands by default. A patch index, `patch_metadata.csv`, lists every tile with its event, bounds, basin, and split.

The split is activation- and basin-exclusive at HydroBASINS Pfafstetter Level 5, so no basin or activation is shared between train, validation, and test, and scores reflect generalization to unseen events.

---

## Building or extending the dataset

The remainder of this README documents the open pipeline that produces the dataset from scratch. Use it to reproduce the release or to extend the dataset to flood activations newer than it. Each processed activation produces 8 analysis-ready GeoTIFFs at mixed resolutions (10 m to 9 km), and Step 6 tiles them into the patches described above.

**Per-activation GeoTIFF outputs (8 files per event: 7 GEE layers + flood mask):**

| File | Bands | Source | GEE Collection |
|---|---|---|---|
| `flood_mask.tif` | 1 (binary: 1=flooded) | Copernicus CEMS flood extent shapefile, 10 m | rasterized from DCC `flood_extent/event.shp` |
| `S1_VV_VH.tif` | 2 (VV, VH) | Sentinel-1 SAR GRD, 10 m | `COPERNICUS/S1_GRD` |
| `S2_NDVI_NDBI.tif` | 2 (NDVI, NDBI) | Sentinel-2 SR, 10 m | `COPERNICUS/S2_SR_HARMONIZED` |
| `MERIT.tif` | 4 (elevation, flow dir, UDA, HAND) | MERIT Hydro, 90 m | `MERIT/Hydro/v1_0_1` |
| `Soil.tif` | 2 (clay, sand) | OpenLandMap SoilGrids, 250 m | `OpenLandMap/SOL/...` |
| `ESA_WorldCover_PermanentWater.tif` | 1 (permanent water mask) | ESA WorldCover, 10 m | `ESA/WorldCover/v200` |
| `Precipitation_{first}_{last}.tif` | N (daily, N days pre-event) | GPM-IMERG V07 daily, ~11 km | `NASA/GPM_L3/IMERG_V07` |
| `SoilMoisture_{first}_{last}.tif` | N (daily, N days pre-event) | SMAP L4 surface SM, ~9 km | `NASA/SMAP/SPL4SMGP/008` |

The seven geospatial layers above are configurable in `scripts/config.py` (`LAYER_TOGGLES` to enable/disable each layer, `N_DAYS_OVERRIDE` to set the daily-series length N per temporal layer, default 30). New GEE layers can be added by copying a template in `scripts/add_gee_layers.py`. `flood_mask.tif` (the rasterized CEMS delineation) is produced in Step 4; the permanent-water layer (`ESA_WorldCover_PermanentWater.tif`, ESA WorldCover) is exported directly from GEE. That gives 8 GeoTIFFs per activation. The temporal layers are written with their antecedent window stamped into the filename, e.g. `Precipitation_20240714_20240812.tif`.

Only two CEMS vector components are used per activation: the AOI boundary (`aoi/aoi.shp`) and the flood extent (`flood_extent/event.shp`). CEMS stopped shipping pre-event hydrography in 2023, so permanent water is taken from ESA WorldCover alone, which is available for every event.

---

## Setup

```bash
conda create -n cems_pipeline python=3.11
conda activate cems_pipeline
pip install -r requirements.txt
```

**GEE authentication (once):**
```bash
earthengine authenticate
```

**Google Drive authentication (once):**  
Two one-time steps, both per Google Cloud project:

1. **Enable the Drive API:** [console.cloud.google.com](https://console.cloud.google.com) → APIs & Services → Enable APIs → search **Google Drive API** → Enable  
   *(Even with credentials.json in place, the API must be explicitly enabled in the project — this is separate from credentials)*
2. **Download OAuth credentials:** APIs & Services → Credentials → Create Credentials → OAuth client ID → Desktop app → Download JSON → place in `Gdrive_credentials/` (any filename is fine)

First run of Script 3 will open a browser for OAuth approval. The token is saved to `data/.gdrive_token.json` — no browser prompt on subsequent runs.

---

## Pipeline

```
config.py                        Edit first: enable/disable layers, set daily-series length
add_gee_layers.py                Layer registry — copy a template here to add a custom GEE layer
1_download_activations.py        Download EMSR flood activations from Copernicus + convert to DCC format
2_submit_gee_tasks.py            Submit GEE export tasks to Google Drive (enabled layers per activation)
                                 # wait for GEE tasks to complete (hours)
3_download_gee_exports.py        Download all EMSR* folders from Google Drive to data/GEE_exports/
4_gee_output_preprocessing.py    Rasterize flood masks + permanent water + build catalog
5_create_splits.py               Assign storm/basin-exclusive train/val/test split
6_make_patches.py                Cut events into model-ready 2.56 km patch tiles
```

```bash
conda activate cems_pipeline
python scripts/1_download_activations.py
python scripts/2_submit_gee_tasks.py
# wait for GEE tasks at code.earthengine.google.com/tasks
python scripts/3_download_gee_exports.py
python scripts/4_gee_output_preprocessing.py
python scripts/5_create_splits.py
python scripts/6_make_patches.py
```

---

## Data layout

```
data/
  activations/
    activations_raw/          raw Copernicus downloads
    activations_dcc/          converted DCC format (aoi/, flood_extent/)
  GEE_exports/
    {EMSR}/{folder_name}/     one folder per activation
      S1_VV_VH.tif                       2 bands  Sentinel-1 VV/VH
      S2_NDVI_NDBI.tif                   2 bands  NDVI + NDBI
      MERIT.tif                          4 bands  elevation, flow direction, UDA, HAND
      Soil.tif                           2 bands  clay + sand (SoilGrids)
      ESA_WorldCover_PermanentWater.tif  1 band   permanent water mask (ESA WorldCover)
      Precipitation_{first}_{last}.tif   N bands  GPM-IMERG daily (N days pre-event)
      SoilMoisture_{first}_{last}.tif    N bands  SMAP daily (N days pre-event)
      flood_mask.tif                     1 band   rasterized CEMS flood extent
  patches/
    {EMSR}/{folder_name}/     2.56 km tiles, 5 GeoTIFFs per patch
      patch_NNNN_input_10m.tif      5 bands   256x256  S1 VV, S1 VH, NDVI, NDBI, permanent water
      patch_NNNN_input_80m.tif      5 bands   32x32    MERIT elev, flowdir sin/cos, UDA, HAND
      patch_NNNN_input_160m.tif     2 bands   16x16    clay, sand
      patch_NNNN_input_2560m.tif    2N bands  1x1      precipitation (N) + soil moisture (N)
      patch_NNNN_flood_mask.tif     1 band    256x256  CEMS flood label
  metadata/
    1_activation_catalog.csv        activation catalog (Script 1)
    1_activation_status.csv         per-product download + DCC status (Script 1)
    2_gee_export_status.csv         per-layer GEE export status (Script 2)
    4_dataset_metadata.csv          final dataset catalog (Script 4)
    4_missing_layers_report.csv     missing enabled layers per activation (Script 4)
    5_split_info.json               split method, counts, exclusivity checks (Script 5)
    patch_metadata.csv              one row per patch tile (Script 6)
    6_patch_validation_issues.csv   per-patch QC findings (Script 6)
```

---

## Dataset CSV

| column | description |
|---|---|
| `folder_name` | activation folder name |
| `region` | europe / rest_of_world |
| `basin_id` | HydroBASINS Pfafstetter Level-5 code |
| `pre_event_sensor` | sensor used for pre-event image |
| `post_event_sensors` | sensor(s) used for post-event image |
| `resolution_post_sensor` | best post-event resolution in metres |
| `resolution_class` | very-high / high / medium |
