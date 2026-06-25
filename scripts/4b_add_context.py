#!/usr/bin/env python3
"""
Script 4b: Add geographic + climate context to the catalog

Adds four per-event columns to released_events_metadata.csv (the Step 4 catalog):

  continent        the continent of the AOI, by point-in-polygon on a world
                   continents layer (Natural Earth), with a nearest-coast
                   fallback and two boundary fixes (see assign_continent).
  climate          Koppen-Geiger main class (A Tropical / B Arid / C Temperate /
                   D Cold / E Polar), the majority class over the AOI sampled from
                   the Beck et al. (2018) raster.
  area_km2         AOI bounding-box area, in the AOI's local UTM zone (km2).
  flood_area_km2   flood_extent polygon area, in the AOI's local UTM zone (km2).

Inputs that are not in the repository (the world continents layer and the Koppen
raster) are downloaded on first run, the same way Step 4 fetches HydroBASINS.

This step runs after Step 4 (which builds the catalog and assigns basin_id) and
before the split (Step 6), which balances on continent.

Input
  data/metadata/released_events_metadata.csv          the catalog (from Step 4)
  data/activations/.../{aoi,flood_extent}/*.shp        the AOI + flood polygons

Output
  data/metadata/released_events_metadata.csv          the four columns added

Usage
  python scripts/4b_add_context.py
"""

import sys
import zipfile
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio
    import requests
    from shapely.geometry import box
except ImportError as e:
    print(f"ERROR: missing dependency ({e}). "
          f"Install with: pip install geopandas rasterio requests")
    sys.exit(1)

import config

CATALOG       = config.CSV_COMPLETE_METADATA
ACTIVATIONS   = config.ACTIVATIONS_DIR
CONTINENTS_DIR = config.CONTINENTS_DIR
CLIMATE_DIR   = config.CLIMATE_DIR

# Natural Earth admin-0 countries (carries a CONTINENT field). ~250 KB.
NE_URL   = ("https://naciscdn.org/naturalearth/110m/cultural/"
            "ne_110m_admin_0_countries.zip")
NE_SHP   = CONTINENTS_DIR / "ne_110m_admin_0_countries.shp"

# Beck et al. (2018) Koppen-Geiger present-day map at 0.0083 deg (~1 km). ~23 MB
# tif inside a small zip on Figshare. Cite Beck et al. (2018) when using it.
KOPPEN_URL = "https://figshare.com/ndownloader/files/12407516"  # Beck_KG_V1.zip
KOPPEN_TIF = CLIMATE_DIR / "Beck_KG_V1_present_0p0083.tif"

# Natural Earth CONTINENT -> our catalog label.
NE_CONTINENT = {
    "Africa": "Africa", "Europe": "Europe", "Asia": "Asia",
    "North America": "North America", "South America": "South America",
    "Oceania": "Australia/Oceania",
}

# Koppen raster value -> main class. A: 1-3, B: 4-7, C: 8-16, D: 17-28, E: 29-30.
KOPPEN_LABEL = {"A": "A Tropical", "B": "B Arid", "C": "C Temperate",
                "D": "D Cold", "E": "E Polar"}


def koppen_main_class(value):
    v = int(value)
    if v <= 0:
        return None
    if v <= 3:
        return "A"
    if v <= 7:
        return "B"
    if v <= 16:
        return "C"
    if v <= 28:
        return "D"
    if v <= 30:
        return "E"
    return None


# ── DOWNLOADS (first run only) ───────────────────────────────────────────────
def _download(url, dest, what):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {what} ...", end=" ", flush=True)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("done")


def ensure_continents():
    if NE_SHP.exists():
        return
    zp = CONTINENTS_DIR / "ne_110m_admin_0_countries.zip"
    _download(NE_URL, zp, "world continents (Natural Earth)")
    with zipfile.ZipFile(zp) as z:
        z.extractall(CONTINENTS_DIR)


def ensure_koppen():
    if KOPPEN_TIF.exists():
        return
    zp = CLIMATE_DIR / "Beck_KG_V1.zip"
    _download(KOPPEN_URL, zp, "Koppen-Geiger raster (Beck et al. 2018)")
    with zipfile.ZipFile(zp) as z:
        z.extractall(CLIMATE_DIR)
    if not KOPPEN_TIF.exists():
        # fall back to whatever present-day 0p0083 tif the archive shipped
        cand = list(CLIMATE_DIR.glob("*present*0p0083*.tif"))
        if cand:
            cand[0].rename(KOPPEN_TIF)


# ── PER-EVENT GEOMETRY HELPERS ───────────────────────────────────────────────
def _aoi_path(folder_name):
    emsr = folder_name.split("_")[0]
    return ACTIVATIONS / emsr / folder_name / "aoi" / "aoi.shp"


def _flood_path(folder_name):
    emsr = folder_name.split("_")[0]
    return ACTIVATIONS / emsr / folder_name / "flood_extent" / "event.shp"


def _local_utm(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def assign_continent(centroid_lonlat, ne):
    """Point-in-polygon continent, nearest-coast fallback, boundary fixes."""
    lon, lat = centroid_lonlat
    pt = gpd.GeoSeries([gpd.points_from_xy([lon], [lat])[0]], crs=4326)
    hit = gpd.sjoin(gpd.GeoDataFrame(geometry=pt, crs=4326),
                    ne[["CONTINENT", "geometry"]], how="left", predicate="within")
    ne_cont = hit["CONTINENT"].iloc[0] if len(hit) else None
    if pd.isna(ne_cont):
        # coastal/ocean centroid: nearest country
        proj = ne.to_crs(3857)
        p = pt.to_crs(3857).iloc[0]
        ne_cont = proj.loc[proj.geometry.distance(p).idxmin(), "CONTINENT"]
    label = NE_CONTINENT.get(ne_cont, ne_cont)
    # Wallacea / SE-Asian islands south of the equator read as Oceania.
    if label == "Asia" and lat < -1 and lon >= 120:
        label = "Australia/Oceania"
    # European Turkey / Thrace (west of ~30 E, north of 40 N) reads as Europe.
    if label == "Asia" and lon <= 30 and lat >= 40:
        label = "Europe"
    return label


def assign_climate(aoi_gdf_4326, koppen):
    """Majority Koppen main class over the AOI footprint."""
    from rasterio.mask import mask as rio_mask
    try:
        out, _ = rio_mask(koppen, [aoi_gdf_4326.union_all()], crop=True)
    except Exception:
        return None
    vals = out[0][out[0] > 0]
    if vals.size == 0:
        return None
    classes = pd.Series([koppen_main_class(v) for v in vals]).dropna()
    if classes.empty:
        return None
    return KOPPEN_LABEL.get(classes.mode().iat[0])


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Script 4b: Add continent / climate / area columns")
    print("=" * 70)

    if not CATALOG.exists():
        print(f"ERROR: catalog not found: {CATALOG}\nRun Step 4 first.")
        sys.exit(1)

    ensure_continents()
    ensure_koppen()
    ne = gpd.read_file(NE_SHP)
    if ne.crs is None or ne.crs.to_epsg() != 4326:
        ne = ne.to_crs(4326)
    koppen = rasterio.open(KOPPEN_TIF)

    cat = pd.read_csv(CATALOG)
    print(f"  catalog events: {len(cat)}")

    continents, climates, areas, flood_areas = [], [], [], []
    for i, fn in enumerate(cat["folder_name"], 1):
        cont = clim = area = flood = None
        ap = _aoi_path(fn)
        if ap.exists():
            g = gpd.read_file(ap)
            if g.crs is not None and g.crs.to_epsg() != 4326:
                g4 = g.to_crs(4326)
            else:
                g4 = g
            c = g4.union_all().centroid
            cont = assign_continent((c.x, c.y), ne)
            clim = assign_climate(g4, koppen)
            utm = _local_utm(c.x, c.y)
            b = g4.to_crs(utm).total_bounds  # AOI bbox in metres
            area = round((b[2] - b[0]) * (b[3] - b[1]) / 1e6, 2)

        fp = _flood_path(fn)
        if fp.exists():
            try:
                fg = gpd.read_file(fp)
                if len(fg) and not fg.union_all().is_empty:
                    fc = (fg.to_crs(4326) if fg.crs and fg.crs.to_epsg() != 4326
                          else fg).union_all().centroid
                    futm = _local_utm(fc.x, fc.y)
                    flood = round(fg.to_crs(futm).area.sum() / 1e6, 2)
            except Exception:
                pass

        continents.append(cont)
        climates.append(clim)
        areas.append(area)
        flood_areas.append(flood)
        if i % 100 == 0:
            print(f"  {i}/{len(cat)}")

    cat["continent"] = continents
    cat["climate"] = climates
    cat["area_km2"] = areas
    cat["flood_area_km2"] = flood_areas
    cat.to_csv(CATALOG, index=False)
    print(f"  wrote continent/climate/area_km2/flood_area_km2 -> {CATALOG.name}")
    print("  continent counts:", cat["continent"].value_counts().to_dict())
    print("  done.")


if __name__ == "__main__":
    main()
