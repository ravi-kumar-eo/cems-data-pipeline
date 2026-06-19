# CEMS Multi-Resolution Flood Dataset

The CEMS Multi-Resolution Flood Dataset is a global, machine-learning-ready dataset for flood mapping. It pairs **463,334** co-registered image patches with observed flood extents from **1,553** Copernicus Emergency Management Service (CEMS) flood events, drawn from **188** rapid-mapping activations between 2017 and 2025 and spanning **281** river basins, six continents, and all five Köppen climate zones.

| Layer | Source | Native resolution | Bands |
|---|---|---|---|
| Sentinel-1 SAR | `COPERNICUS/S1_GRD` | 10 m | VV, VH |
| Sentinel-2 indices | `COPERNICUS/S2_SR_HARMONIZED` | 10 m | NDVI, NDBI |
| MERIT Hydro | `MERIT/Hydro/v1_0_1` | 90 m | elevation, flow direction, UDA, HAND |
| SoilGrids | `OpenLandMap/SOL` | 250 m | clay %, sand % |
| ESA WorldCover | `ESA/WorldCover/v200` | 10 m | permanent-water mask |
| Precipitation | `NASA/GPM_L3/IMERG_V07` | ~11 km | 30 daily (mm/day) |
| Soil moisture | `NASA/SMAP/SPL4SMGP/008` | ~9 km | 30 daily (m³/m³) |
| Flood label | CEMS flood extent (`event.shp`) | 10 m | inundation mask (1 = flooded) |

Every input precedes the flood, so the dataset poses flood **prediction** from antecedent conditions rather than post-event mapping. It is released as patch tiles with the train, validation, and test split already assigned, ready to load for model training. The full pipeline that produces the patches is included, so the release can be rebuilt from scratch or extended to new flood activations with a Google Earth Engine account.

**Dataset:** [Zenodo DOI to be added]

## Dataset description

The dataset is delivered as patches. Each flood event is cut into square, non-overlapping tiles that each cover a 2.56 km × 2.56 km ground footprint. **One patch is five GeoTIFFs: four input files and one flood-label file.** The four input files hold the layers above, grouped by resolution, and the label file holds the CEMS flood mask.

| File | Bands | Size | Contents |
|---|---|---|---|
| `input_10m.tif` | 5 | 256×256 | S1 VV, S1 VH, NDVI, NDBI, permanent water |
| `input_80m.tif` | 5 | 32×32 | MERIT elevation, flow-dir sin, flow-dir cos, UDA, HAND |
| `input_160m.tif` | 2 | 16×16 | SoilGrids clay %, sand % |
| `input_2560m.tif` | 2N | 1×1 | precipitation (N days), soil moisture (N days) |
| `flood_mask.tif` | 1 | 256×256 | flood label (1 = flooded) |

Only the 10 m layers are kept at their native resolution, as a 256×256 grid. The other layers are resampled so they integrate into a single multi-modal stack: each file covers the same 2.56 km × 2.56 km footprint, sampled to the grid that matches its resolution. Precipitation and soil moisture reduce to one cell per tile, one value per antecedent day, so `input_2560m` holds 2N bands. The released dataset uses 30 antecedent days, giving 60 bands, 30 precipitation days followed by 30 soil-moisture days. The number of days N is configurable in the pipeline (Section below), so a newly prepared dataset can use a different window.

The permanent-water band lets a model tell pre-existing water from new flooding, while the label stays the observed CEMS inundation alone. MERIT flow direction is split into the sine and cosine of its compass angle so the circular variable has no discontinuity.

### Patch index and splits

Every patch is listed in `data/metadata/patch_metadata.csv`, one row per tile. The three split files, `train_patches.csv`, `val_patches.csv`, and `test_patches.csv`, are the same table filtered by the `split` column, so each can be loaded directly as a training, validation, or test set. The split is assigned at HydroBASINS Pfafstetter Level 5 and is exclusive by basin and by activation, so no basin and no activation crosses the train, validation, and test sets.

A tile is addressed by `(emsr_code, folder_name, patch_number)`, which locate its files on disk, so the CSV references patches relationally rather than by absolute path. The key columns are below.

| Column | Description |
|---|---|
| `patch_index` | global running index over all patches |
| `emsr_code` | Copernicus activation code, e.g. `EMSR203` |
| `folder_name` | event the patch belongs to |
| `patch_number` | index of the tile within its event (the `NNNN` in the filenames) |
| `crs` | coordinate reference system of the tile (per-event UTM zone) |
| `bounds_minx/miny/maxx/maxy` | tile footprint bounds in `crs` units (m) |
| `transform_a..f` | affine geo-transform of the 10 m grid |
| `resolution_class` | post-event sensor class: medium, high, or very-high |
| `basin_id` | HydroBASINS Pfafstetter Level-5 code(s) of the event |
| `continent` | continent of the event |
| `flood_fraction` | fraction of flooded pixels in the tile, from the flood mask |
| `split` | train, val, or test |

---

## Building or extending the dataset

The rest of this README documents the open pipeline that builds the dataset from scratch. Use it to reproduce the release or to extend it to newer activations. The pipeline produces the eight per-event layers of the overview table (the seven geospatial layers plus the flood mask) as full-scene GeoTIFFs, then Step 6 tiles them into the patches described above.

The seven geospatial layers are configurable in `scripts/config.py`. `LAYER_TOGGLES` enables or disables each layer, and `N_DAYS_OVERRIDE` sets the daily-series length N for the temporal layers (default 30). New GEE layers can be added by copying a template in `scripts/add_gee_layers.py`. The full-scene files keep their own names: `S1_VV_VH.tif`, `S2_NDVI_NDBI.tif`, `MERIT.tif`, `Soil.tif`, `ESA_WorldCover_PermanentWater.tif`, and the temporal layers carry their antecedent window in the filename, for example `Precipitation_20240714_20240812.tif`. `flood_mask.tif` is produced in Step 4 by rasterizing the CEMS delineation.

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

The released catalog is `complete_dataset_metadata.csv`, one row per event across the whole dataset. Its `folder_name` keys into the patches, GEE_exports, and activations_reorganized folders. A pipeline run does not rewrite it; Step 4 writes only the events new in that run to `4_dataset_metadata.csv` and appends them to the released catalog, so prior events and their assigned splits are preserved.

The columns are below.

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
