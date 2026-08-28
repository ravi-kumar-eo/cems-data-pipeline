#!/usr/bin/env python3
"""
gee_direct.py — direct (Drive-free) GEE image download.

Fetches an ee.Image straight to a local GeoTIFF with tiled getDownloadURL
requests, so Script 2 can write layers into data/GEE_exports/ immediately and
Script 3 (Google Drive) becomes unnecessary. Grid semantics are identical to
the Drive export path: the caller passes the same crs_transform and snapped
bounds it would have given Export.image.toDrive, and the result lands on
exactly that grid.

Tiles are square, sized per band count (see tile_size), fetched by MAX_WORKERS
threads with retries, then stitched and written atomically (tmp file + rename) as LZW-compressed GeoTIFF.
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

# Tile side in pixels. GEE caps an interactive getDownloadURL at 50 MB, and it
# sizes the request as width*height*bands*8 (float64 internally, whatever the
# band type), so the largest legal square depends on the BAND COUNT:
#
#     2 bands (S1, S2)  -> 1664 px  (44.3 MB)
#     4 bands (MERIT)   -> 1152 px  (42.5 MB)
#    30 bands (weather) ->  384 px  (35.4 MB)
#
# Bigger tiles are markedly cheaper per pixel — one 1664 px 2-band tile takes
# ~17 s versus ~14.6 s for a 1024 px tile covering 2.6x less ground, i.e. 2.2x
# faster per pixel — because each request carries a fixed GEE-side rendering
# cost. TILE_MAX caps it so a single failed tile never wastes too much work.
TILE_BYTES_LIMIT = 50 * 1024 * 1024
TILE_SAFETY = 0.90          # stay clear of the hard limit
TILE_MAX = 1664
TILE_MIN = 256


def tile_size(n_bands: int) -> int:
    """Largest square tile (px) whose getDownloadURL request stays under the cap."""
    budget = TILE_BYTES_LIMIT * TILE_SAFETY / max(1, n_bands) / 8
    side = int(budget ** 0.5)
    side -= side % 64                       # keep it a tidy multiple of 64
    return max(TILE_MIN, min(TILE_MAX, side))


# Wall-clock cap for the tiled-download phase (as_completed timeout).
#
# The budget SCALES WITH THE TILE COUNT: a fixed cap silently truncates large
# AOIs, because the whole layer is abandoned when the clock runs out no matter
# how many tiles were still in flight. A 26126x30534 px AOI is 780 tiles, which
# cannot finish inside a flat 600 s at MAX_WORKERS=3 — that is how several
# events ended up with S1/S2 (and therefore flood_mask) covering only a corner
# of their AOI. Budget = SECONDS_PER_TILE per tile per worker, floored at
# LAYER_DEADLINE_MIN so small layers still fail fast on a genuinely wedged tile.
#
# SECONDS_PER_TILE is the MEASURED SUSTAINED rate — wall-clock seconds per tile
# for the whole layer, ALREADY NET OF PARALLELISM. Do not divide it by the
# worker count: 24 tiles at 1664 px with 8 workers took 254 s = 10.6 s/tile
# sustained, versus ~4 s/tile in a short burst, because GEE throttles sustained
# concurrent rendering. Budgeting from the burst rate under-budgets ~4x and the
# layer is abandoned half-finished. 16 s carries the slow tail over 10.6 s.
LAYER_DEADLINE_MIN = 600
SECONDS_PER_TILE = 16.0
LAYER_DEADLINE_MAX = 21600          # 6 h — a real hang, not a big AOI
# Extra head-room for the SIGALRM backstop covering the WHOLE download_image
# call, including bandNames().getInfo() and any getDownloadURL that hangs BEFORE
# the executor — those are outside the as_completed timeout and would otherwise
# wedge the run forever. Larger than the layer deadline so the finer
# as_completed timeout always wins first.
HARD_DEADLINE_MARGIN = 90


def layer_deadline(n_tiles: int) -> int:
    """
    Wall-clock budget for a tiled layer download, in seconds.

    SECONDS_PER_TILE is a sustained, post-parallelism rate, so the estimate is
    simply rate x tiles — no division by the worker count.
    """
    est = SECONDS_PER_TILE * n_tiles
    return int(min(LAYER_DEADLINE_MAX, max(LAYER_DEADLINE_MIN, est)))


# 8 parallel tile requests. Measured on a real AOI: 1 worker 19.6 s/tile,
# 4 workers 5.3 s/tile, 8 workers 4.2 s/tile effective — GEE-side rendering
# dominates, so concurrency scales well. 8 stays inside GEE's request limits.
MAX_WORKERS = 8
RETRIES = 5


class _HardTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _HardTimeout()


def dims_from_bounds(minx: float, miny: float, maxx: float, maxy: float,
                     pixel: float):
    """Width/height in pixels of a snapped bbox on a pixel-deg grid."""
    return (int(round((maxx - minx) / pixel)),
            int(round((maxy - miny) / pixel)))


def _fetch_tile(image, bands, tf: list, col0: int, row0: int, w: int, h: int):
    """One tiled getDownloadURL request; returns (col0, row0, array)."""
    tile_tf = [tf[0], tf[1], tf[2] + col0 * tf[0],
               tf[3], tf[4], tf[5] + row0 * tf[4]]
    # getDownloadURL() must be INSIDE the retry loop: it is itself a server call
    # and returns 503 when GEE is busy. Retrying only urlopen() leaves that case
    # unhandled, and one 503 then kills a layer that is 20 minutes in — the URL
    # is also short-lived, so a stale one cannot be reused across a long backoff.
    for attempt in range(RETRIES):
        try:
            url = image.select(bands).getDownloadURL({
                "crs": "EPSG:4326", "crs_transform": tile_tf,
                "dimensions": [w, h], "format": "GEO_TIFF"})
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
    # Band count decides the legal tile size, so resolve it before tiling.
    if band_names is None:
        band_names = image.bandNames().getInfo()
    tile = tile_size(len(band_names))
    jobs = [(c, r, min(tile, width - c), min(tile, height - r))
            for r in range(0, height, tile) for c in range(0, width, tile)]
    deadline = layer_deadline(len(jobs))

    # SIGALRM backstop — main-thread only, which is where the caller invokes this.
    # Guarantees the call returns within the budget even if getInfo /
    # getDownloadURL hang before the executor's own as_completed timeout applies.
    have_alarm = hasattr(signal, "SIGALRM")
    prev_handler = signal.signal(signal.SIGALRM, _on_alarm) if have_alarm else None
    if have_alarm:
        signal.alarm(deadline + HARD_DEADLINE_MARGIN)
    try:
        img = image.unmask(NODATA)  # only fills still-masked pixels
        if len(jobs) > 200:
            print(f"        ({len(jobs)} tiles @{tile}px, budget {deadline // 60} min)")
        arr = np.full((len(band_names), height, width), NODATA, dtype=np.float32)
        ex = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        futs = [ex.submit(_fetch_tile, img, band_names, crs_transform, *j)
                for j in jobs]
        try:
            for fut in as_completed(futs, timeout=deadline):
                col0, row0, data = fut.result()
                arr[:, row0:row0 + data.shape[1],
                    col0:col0 + data.shape[2]] = data
            ex.shutdown(wait=True)
        except FuturesTimeout:
            # A tile is wedged (hung getDownloadURL / stalled socket). Abandon
            # without waiting so the caller can move on; leaked threads unwind
            # on their own once their urlopen timeout finally fires.
            ex.shutdown(wait=False, cancel_futures=True)
            print(f"        ✗ direct download timed out after {deadline}s "
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
        print(f"        ✗ direct download hard-timeout after "
              f"{deadline + HARD_DEADLINE_MARGIN}s "
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
