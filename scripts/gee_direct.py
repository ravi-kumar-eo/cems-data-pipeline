#!/usr/bin/env python3
"""
gee_direct.py — direct (Drive-free) GEE image download.

Fetches an ee.Image straight to a local GeoTIFF with tiled getDownloadURL
requests, so Script 2 can write layers into data/GEE_exports/ immediately and
Script 3 (Google Drive) becomes unnecessary. Grid semantics are identical to
the Drive export path: the caller passes the same crs_transform and snapped
bounds it would have given Export.image.toDrive, and the result lands on
exactly that grid.

Tiles are TILE px squares fetched by MAX_WORKERS threads with retries, then
stitched and written atomically (tmp file + rename) as LZW-compressed GeoTIFF.
Interactive getDownloadURL requests are capped at ~50 MB each, hence tiling.
"""
import io
import time
import signal
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

NODATA = -9999.0
TILE = 1024
# Wall-clock cap for the tiled-download phase (as_completed timeout).
LAYER_DEADLINE = 600
# Absolute SIGALRM backstop covering the WHOLE download_image call, including the
# bandNames().getInfo() and any getDownloadURL that hang BEFORE the executor —
# those are outside LAYER_DEADLINE and would otherwise wedge the run forever.
# Slightly larger than LAYER_DEADLINE so the finer as_completed timeout wins first.
HARD_DEADLINE = LAYER_DEADLINE + 90


class _HardTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _HardTimeout()
MAX_WORKERS = 3
RETRIES = 5


def dims_from_bounds(minx: float, miny: float, maxx: float, maxy: float,
                     pixel: float):
    """Width/height in pixels of a snapped bbox on a pixel-deg grid."""
    return (int(round((maxx - minx) / pixel)),
            int(round((maxy - miny) / pixel)))


def _fetch_tile(image, bands, tf: list, col0: int, row0: int, w: int, h: int):
    """One tiled getDownloadURL request; returns (col0, row0, array)."""
    tile_tf = [tf[0], tf[1], tf[2] + col0 * tf[0],
               tf[3], tf[4], tf[5] + row0 * tf[4]]
    url = image.select(bands).getDownloadURL({
        "crs": "EPSG:4326", "crs_transform": tile_tf,
        "dimensions": [w, h], "format": "GEO_TIFF"})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                buf = r.read()
            with rasterio.open(io.BytesIO(buf)) as src:
                return col0, row0, src.read()
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(15 * (attempt + 1))


def download_image(image, crs_transform: list, width: int, height: int,
                   out_path: Path, band_names=None) -> bool:
    """
    Download `image` onto the (crs_transform, width, height) EPSG:4326 grid
    and write it to out_path. Returns True on success, False on failure
    (partial tmp files are removed; a failed layer can simply be retried).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.tif")
    ex = None
    # SIGALRM backstop — main-thread only, which is where the caller invokes this.
    # Guarantees the call returns within HARD_DEADLINE even if getInfo /
    # getDownloadURL hang before the executor's own as_completed timeout applies.
    have_alarm = hasattr(signal, "SIGALRM")
    prev_handler = signal.signal(signal.SIGALRM, _on_alarm) if have_alarm else None
    if have_alarm:
        signal.alarm(HARD_DEADLINE)
    try:
        if band_names is None:
            band_names = image.bandNames().getInfo()
        img = image.unmask(NODATA)  # only fills still-masked pixels
        jobs = [(c, r, min(TILE, width - c), min(TILE, height - r))
                for r in range(0, height, TILE) for c in range(0, width, TILE)]
        arr = np.full((len(band_names), height, width), NODATA, dtype=np.float32)
        ex = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        futs = [ex.submit(_fetch_tile, img, band_names, crs_transform, *j)
                for j in jobs]
        try:
            for fut in as_completed(futs, timeout=LAYER_DEADLINE):
                col0, row0, data = fut.result()
                arr[:, row0:row0 + data.shape[1],
                    col0:col0 + data.shape[2]] = data
            ex.shutdown(wait=True)
        except FuturesTimeout:
            # A tile is wedged (hung getDownloadURL / stalled socket). Abandon
            # without waiting so the caller can move on; leaked threads unwind
            # on their own once their urlopen timeout finally fires.
            ex.shutdown(wait=False, cancel_futures=True)
            print(f"        ✗ direct download timed out after {LAYER_DEADLINE}s "
                  f"(layer abandoned)")
            tmp.unlink(missing_ok=True)
            return False
        tf = Affine(*crs_transform[:2], crs_transform[2],
                    *crs_transform[3:5], crs_transform[5])
        prof = dict(driver="GTiff", dtype="float32", count=len(band_names),
                    width=width, height=height, crs="EPSG:4326", transform=tf,
                    nodata=NODATA, compress="lzw", tiled=True,
                    bigtiff="if_safer")
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(arr)
            dst.descriptions = tuple(band_names)
        tmp.rename(out_path)
        return True
    except _HardTimeout:
        # A call hung before/around the executor (e.g. getInfo). Abandon so the
        # caller moves on instead of the whole pipeline wedging on one activation.
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)
        print(f"        ✗ direct download hard-timeout after {HARD_DEADLINE}s "
              f"(pre-download hang, abandoned)")
        tmp.unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f"        ✗ direct download failed: {e}")
        tmp.unlink(missing_ok=True)
        return False
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)
