# CEMS Multi-Resolution Flood Dataset

The CEMS Multi-Resolution Flood Dataset is a global, machine-learning-ready dataset for flood mapping. It pairs **463,334** co-registered image patches with observed flood extents from **1,553** Copernicus Emergency Management Service (CEMS) flood events, drawn from **188** rapid-mapping activations between 2017 and 2025 and spanning **281** river basins, six continents, and all five Köppen climate zones.

Each patch stacks four input groups at their native resolutions — 10 m Sentinel-1 SAR and Sentinel-2 indices, 80 m MERIT terrain, 160 m SoilGrids, and a 2.56 km cell of daily pre-event precipitation and soil moisture — against a 10 m flood label. The radar is acquired *before* the flood, so the task the dataset poses is flood **prediction** from antecedent conditions, not post-event mapping. The weather layers carry a 30-day window of daily precipitation and soil moisture leading up to each event, giving a model the rainfall and wetness history that drives inundation.

The patches ship ready for training with a train, validation, and test split already fixed at HydroBASINS Pfafstetter Level 5, exclusive by basin and by activation so that no basin and no activation crosses splits. The full reproduction pipeline is included, so the release can be rebuilt from scratch or extended to new activations with a Google Earth Engine account.

**Dataset:** [Zenodo DOI to be added]

## The patch dataset

Each of the 1,553 flood events is cut into square, non-overlapping 2.56 km tiles. A tile is five co-registered GeoTIFFs: four input groups at their own resolution and the flood label. The weather layers cover `N` days before the flood, 30 by default, so the bands below are written in terms of `N`.

| File | Bands | Pixels | Contents |
|---|---|---|---|
| `patch_NNNN_input_10m.tif` | 5 | 256×256 | S1 VV, S1 VH, NDVI, NDBI, permanent water |
| `patch_NNNN_input_80m.tif` | 5 | 32×32 | MERIT elevation, flow-dir sin, flow-dir cos, UDA, HAND |
| `patch_NNNN_input_160m.tif` | 2 | 16×16 | SoilGrids clay %, sand % |
| `patch_NNNN_input_2560m.tif` | 2N | 1×1 | precipitation (N days) then soil moisture (N days) |
| `patch_NNNN_flood_mask.tif` | 1 | 256×256 | flood label (1 = flooded) |

With the default `N` of 30, `input_2560m` holds 60 bands: precipitation for days 1–30, then soil moisture for days 1–30. Every file shares the same 2.56 km footprint at its own resolution; the weather layers are coarser than a tile, so `input_2560m` is a single 2.56 km cell.

The flood label is the CEMS flood delineation. Permanent water is a separate input band (band 5 of `input_10m`), taken from ESA WorldCover, so a model can tell pre-existing water from new flooding while the label stays the observed inundation. MERIT flow direction is split into the sine and cosine of its compass angle. The patch index `patch_metadata.csv` lists every tile with its event, bounds, basin, and split.

To train on the dataset, download the patches from the Zenodo link above. To reproduce or extend it, run the pipeline below.

---

## Building or extending the dataset

The rest of this README documents the open pipeline that builds the dataset from scratch — use it to reproduce the release or to extend it to newer activations. The pipeline turns each event into 8 analysis-ready GeoTIFFs at mixed resolutions (10 m to 9 km), then Step 6 tiles them into the patches described above.

**Per-event GeoTIFF outputs (8 files per event: 7 GEE layers + flood mask):**

| File | Bands | Source | GEE Collection |
|---|---|---|---|
| `flood_mask.tif` | 1 (binary: 1=flooded) | Copernicus CEMS flood extent shapefile, 10 m | rasterized from `flood_extent/event.shp` |
| `S1_VV_VH.tif` | 2 (VV, VH) | Sentinel-1 SAR GRD, 10 m | `COPERNICUS/S1_GRD` |
| `S2_NDVI_NDBI.tif` | 2 (NDVI, NDBI) | Sentinel-2 SR, 10 m | `COPERNICUS/S2_SR_HARMONIZED` |
| `MERIT.tif` | 4 (elevation, flow dir, UDA, HAND) | MERIT Hydro, 90 m | `MERIT/Hydro/v1_0_1` |
| `Soil.tif` | 2 (clay, sand) | OpenLandMap SoilGrids, 250 m | `OpenLandMap/SOL/...` |
| `ESA_WorldCover_PermanentWater.tif` | 1 (permanent water mask) | ESA WorldCover, 10 m | `ESA/WorldCover/v200` |
| `Precipitation_{first}_{last}.tif` | N (daily, N days pre-event) | GPM-IMERG V07 daily, ~11 km | `NASA/GPM_L3/IMERG_V07` |
| `SoilMoisture_{first}_{last}.tif` | N (daily, N days pre-event) | SMAP L4 surface SM, ~9 km | `NASA/SMAP/SPL4SMGP/008` |

The seven geospatial layers above are configurable in `scripts/config.py`. `LAYER_TOGGLES` enables or disables each layer, and `N_DAYS_OVERRIDE` sets the daily-series length N per temporal layer (default 30). New GEE layers can be added by copying a template in `scripts/add_gee_layers.py`. `flood_mask.tif` is produced in Step 4 by rasterizing the CEMS delineation. The permanent-water layer (`ESA_WorldCover_PermanentWater.tif`) is exported directly from GEE. That gives 8 GeoTIFFs per event. The temporal layers carry their antecedent window in the filename, for example `Precipitation_20240714_20240812.tif`.

Each activation supplies two CEMS vector components: the AOI boundary (`aoi/aoi.shp`) and the flood extent (`flood_extent/event.shp`). Permanent water comes from ESA WorldCover, which covers every event.

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
   *(Even with credentials.json in place, the API must be explicitly enabled in the project. This is separate from credentials.)*
2. **Download OAuth credentials:** APIs & Services → Credentials → Create Credentials → OAuth client ID → Desktop app → Download JSON → place in `Gdrive_credentials/` (any filename is fine)

First run of Script 3 will open a browser for OAuth approval. The token is saved to `data/.gdrive_token.json`, so subsequent runs do not prompt.

---

## Pipeline

```
config.py                        Edit first: enable/disable layers, set daily-series length
add_gee_layers.py                Layer registry. Copy a template here to add a custom GEE layer
1_download_activations.py        Download EMSR flood activations from Copernicus + reorganize into standardized folders
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
    activations_reorganized/  standardized shapefiles (aoi/, flood_extent/)
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
      patch_NNNN_input_2560m.tif    2N bands  1x1      precipitation (N days) then soil moisture (N days)
      patch_NNNN_flood_mask.tif     1 band    256x256  CEMS flood label
  metadata/
    1_activation_catalog.csv        activation catalog (Script 1)
    1_activation_status.csv         per-product download + reorganization status (Script 1)
    2_gee_export_status.csv         per-layer GEE export status (Script 2)
    4_dataset_metadata.csv          events new in the latest run (Script 4)
    complete_dataset_metadata.csv   full accumulated dataset catalog (Script 4)
    4_missing_layers_report.csv     missing enabled layers per activation (Script 4)
    5_split_info.json               split method, counts, exclusivity checks (Script 5)
    patch_metadata.csv              one row per patch tile (Script 6)
    6_patch_validation_issues.csv   per-patch QC findings (Script 6)
```

---

## Dataset catalog

`complete_dataset_metadata.csv` has one row per event. Its `folder_name` keys into the patches, GEE_exports, and activations_reorganized folders.

| column | description |
|---|---|
| `folder_name` | event folder name |
| `basin_id` | HydroBASINS Pfafstetter Level-5 code(s) |
| `pre_event_sensor` | sensor of the pre-event reference image |
| `post_event_sensors` | sensor(s) used for the flood delineation |
| `resolution_post_sensor` | post-event sensor resolution (m) |
| `resolution_class` | medium, high, or very-high |
| `continent` | continent of the area of interest |
| `climate` | Köppen-Geiger main class |
| `split` | train, validation, or test |
| `area_km2` | area of interest size (km²) |
| `flood_area_km2` | area under water from the flood polygon (km²) |
