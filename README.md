# CEMS Flood Dataset Pipeline

A pipeline to build a multi-modal flood dataset from Copernicus EMSR activations and Google Earth Engine satellite exports.

**What it produces:** One folder per flood event containing 7 stacked GeoTIFFs (Sentinel-1, Sentinel-2 indices, MERIT DEM, soil, ESA permanent water, ERA5 precipitation, SMAP soil moisture).

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
                              ── wait for GEE tasks to complete (hours) ──
3_download_gee_exports.py     Download all EMSR* folders from Google Drive → data/GEE_exports/
4_validate_exports.py         Validate exports + build flood_dataset.csv
```

```bash
conda activate cems_pipeline
python scripts/1_download_activations.py
python scripts/2_submit_gee_tasks.py
# wait for GEE tasks at code.earthengine.google.com → Tasks
python scripts/3_download_gee_exports.py
python scripts/4_validate_exports.py
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
  activations.csv             first-draft catalog from Script 1
  flood_dataset.csv           final dataset catalog from Script 4
metadata/
  flood_dataset.csv           archival copy of the final catalog
```

---

## Dataset CSV (`flood_dataset.csv`)

| column | description |
|---|---|
| `folder_name` | activation folder name |
| `region` | europe / rest_of_world |
| `basin_id` | HydroSHEDS HydroBASINS level 12 ID |
| `pre_event_sensor` | sensor used for pre-event image |
| `post_event_sensors` | sensor(s) used for post-event image |
| `resolution_post_sensor` | best post-event resolution in metres |
| `resolution_class` | very-high / high / medium |
