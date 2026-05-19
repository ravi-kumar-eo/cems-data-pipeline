# CEMS Flood Dataset Pipeline

A pipeline to build a multi-resolution flood dataset from Copernicus EMSR rapid mapping activations and EO data from GEE. Covers the full workflow from raw CEMS product download and DCC conversion to GEE export submission, Drive retrieval, and final dataset validation. Each processed activation produces 7 analysis-ready GeoTIFFs at mixed resolutions (10 m to 9 km), aligned to a common grid.

**Per-activation GeoTIFF exports (7 files per event):**

| File | Bands | Source | GEE Collection |
|---|---|---|---|
| `S1_VV_VH.tif` | 2 (VV, VH) | Sentinel-1 SAR GRD, 10 m | `COPERNICUS/S1_GRD` |
| `land_cover.tif` | 2 (NDVI, NDBI) | Sentinel-2 SR, 10 m | `COPERNICUS/S2_SR_HARMONIZED` |
| `MERIT.tif` | 4 (elevation, flow dir, UDA, HAND) | MERIT Hydro, 90 m | `MERIT/Hydro/v1_0_1` |
| `Soil.tif` | 2 (clay, sand) | OpenLandMap SoilGrids, 250 m | `OpenLandMap/SOL/...` |
| `ESA_PW.tif` | 1 (permanent water mask) | ESA WorldCover, 10 m | `ESA/WorldCover/v200` |
| `Precipitation.tif` | 10 (daily, 10 days pre-event) | ERA5-Land daily, 9 km | `ECMWF/ERA5_LAND/DAILY_AGGR` |
| `SoilMoisture.tif` | 10 (daily, 10 days pre-event) | SMAP 10 km, 10 km | `NASA_USDA/HSL/SMAP10KM_soil_moisture` |

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
Enable the Drive API and download `credentials.json`:
1. [console.cloud.google.com](https://console.cloud.google.com) → APIs & Services → Enable APIs → search **Google Drive API** → Enable
2. APIs & Services → Credentials → Create Credentials → OAuth client ID → Desktop app → Download JSON
3. Rename to `credentials.json` and place it in this repo root

First run of Script 3 will open a browser for OAuth approval.

---

## Pipeline

```
1_download_activations.py     Download EMSR flood activations from Copernicus + convert to DCC format
2_submit_gee_tasks.py         Submit GEE export tasks to Google Drive (7 layers per activation)
                              # wait for GEE tasks to complete (hours)
3_download_gee_exports.py     Download all EMSR* folders from Google Drive to data/GEE_exports/
4_check_exports.py            Validate exports + build dataset_metadata.csv
```

```bash
conda activate cems_pipeline
python scripts/1_download_activations.py
python scripts/2_submit_gee_tasks.py
# wait for GEE tasks at code.earthengine.google.com/tasks
python scripts/3_download_gee_exports.py
python scripts/4_check_exports.py
```

---

## Data layout

```
data/
  activations/
    activations_raw/          raw Copernicus downloads
    activations_dcc/          converted DCC format (aoi/, flood_extent/, permanent_water/)
  GEE_exports/
    {folder_name}/            one folder per activation
      S1_VV_VH.tif            2 bands  Sentinel-1 VV/VH
      land_cover.tif          2 bands  NDVI + NDBI
      MERIT.tif               4 bands  elevation, flow direction, UDA, HAND
      Soil.tif                2 bands  clay + sand (SoilGrids)
      ESA_PW.tif              1 band   permanent water mask
      Precipitation.tif      10 bands  ERA5-Land daily (10 days pre-event)
      SoilMoisture.tif       10 bands  SMAP daily (10 days pre-event)
metadata/
  activations.csv             activation catalog (Script 1)
  activations_status.csv      per-product download + DCC status (Script 1)
  gee_tasks_record.csv        GEE task tracking (Script 2)
  dataset_metadata.csv           final dataset catalog (Script 4)
```

---

## Dataset CSV

| column | description |
|---|---|
| `folder_name` | activation folder name |
| `region` | europe / rest_of_world |
| `basin_id` | HydroSHEDS HydroBASINS level 12 ID |
| `pre_event_sensor` | sensor used for pre-event image |
| `post_event_sensors` | sensor(s) used for post-event image |
| `resolution_post_sensor` | best post-event resolution in metres |
| `resolution_class` | very-high / high / medium |
