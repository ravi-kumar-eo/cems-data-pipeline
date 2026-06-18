#!/usr/bin/env python3
"""
Script 5: Create Train/Val/Test Splits

Assigns every activation (event) in dataset_metadata.csv to a train/val/test
split and writes the assignment back as a new `split` column. The split is
activation + basin exclusive at HydroBASINS Pfafstetter Level 5:

  - activation = EMSR event code (e.g. EMSR866). One activation can map a flood
                 across many basins, so a basin-only split would let the same
                 activation appear in two splits through different basins.
  - basin      = HydroBASINS Pfafstetter Level-5 code (first 5 digits of PFAF_ID),
                 looked up from each event's Level-12 basin_id.

Events are grouped by the connected components of the (basin <-> activation) graph,
so neither a basin nor an activation ever crosses a split boundary. Whole
components are then assigned to train/val/test with a greedy that targets
70/15/15 by event count while balancing the three resolution classes.

Outputs:
  data/metadata/4_dataset_metadata.csv   <- rewritten with `split` column
  data/metadata/5_split_info.json        <- method, counts, ratios, overlap checks
  data/metadata/5_basin_split_map.png    <- Level-5 basins coloured by split

Input:
  data/metadata/4_dataset_metadata.csv   (from Script 4)
  data/hydrobasins/hybas_lev12_global.*  (HYBAS_ID -> PFAF_ID lookup + geometry)
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR     = _SCRIPTS_DIR.parent
DATA_DIR     = BASE_DIR / "data"
META_DIR     = DATA_DIR / "metadata"

import config
DATASET_METADATA_CSV = config.CSV_DATASET_METADATA
HYDROBASINS_SHP      = DATA_DIR / "hydrobasins" / "hybas_lev12_global.shp"
SPLIT_INFO_JSON      = config.JSON_SPLIT_INFO
SPLIT_MAP_PNG        = META_DIR / "5_basin_split_map.png"

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASIN_LEVEL   = 5
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RES_CLASSES   = ["medium", "high", "very-high"]
SPLIT_COLORS  = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca44e"}


# ── LEVEL-5 BASIN CODES ──────────────────────────────────────────────────────
def basin_l5_for_event(basin_id):
    """
    Return the set of Level-5 basin codes for an event. Script 4 already stores
    basin_id as a Pfafstetter Level-5 code (or dash-joined codes for multi-basin
    events), so this just splits and normalises to 5 digits.
    """
    codes = set()
    for code in str(basin_id).split("-"):
        code = code.strip()
        if code and code.lower() != "nan":
            codes.add(code[:BASIN_LEVEL])
    return codes


# ── CONNECTED COMPONENTS (union-find over basins + activations) ───────────────
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ── GREEDY SPLIT ASSIGNMENT ───────────────────────────────────────────────────
def assign_components(components, totals):
    """
    Assign whole components to train/val/test.

    components: list of dicts with 'events' (list) and 'res_counts' (dict).
    totals:     {'n': N, 'medium': .., 'high': .., 'very-high': ..} global counts.
    Returns {split: [event_rows]}.

    For each component (largest first) pick the split with the largest summed
    relative deficit across total-event-count and the three resolution classes,
    so both the overall ratio and the resolution mix stay balanced.
    """
    targets = {s: {"n": TARGET_RATIOS[s] * totals["n"],
                   **{r: TARGET_RATIOS[s] * totals[r] for r in RES_CLASSES}}
               for s in TARGET_RATIOS}
    current = {s: {"n": 0, **{r: 0 for r in RES_CLASSES}} for s in TARGET_RATIOS}
    result  = {s: [] for s in TARGET_RATIOS}

    def deficit(s, comp):
        d = 0.0
        for key in ["n"] + RES_CLASSES:
            tgt = targets[s][key]
            if tgt <= 0:
                continue
            cur = current[s][key]
            add = len(comp["events"]) if key == "n" else comp["res_counts"].get(key, 0)
            # reward filling an under-target split, penalise overshoot
            d += (tgt - cur) / tgt - max(0, (cur + add) - tgt) / tgt
        return d

    ordered = sorted(components, key=lambda c: (-len(c["events"]), c["key"]))
    for comp in ordered:
        best = max(TARGET_RATIOS, key=lambda s: (deficit(s, comp), s == "train"))
        result[best].extend(comp["events"])
        current[best]["n"] += len(comp["events"])
        for r in RES_CLASSES:
            current[best][r] += comp["res_counts"].get(r, 0)
    return result


# ── BASIN SPLIT MAP ───────────────────────────────────────────────────────────
def make_split_map(event_basins, event_split):
    """Choropleth of the Level-5 basins present, coloured by their split."""
    try:
        import warnings; warnings.filterwarnings("ignore")
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import cartopy.io.shapereader as shpreader

        basin_split = {}
        for ev, basins in event_basins.items():
            for b in basins:
                basin_split.setdefault(b, event_split[ev])  # first writer wins
        present = set(basin_split)
        if not present:
            print("  ! No basins to map"); return

        print(f"  Reading basin geometry for {len(present)} Level-5 basins ...")
        gdf = gpd.read_file(HYDROBASINS_SHP)
        gdf["l5"] = gdf["PFAF_ID"].astype("int64").astype(str).str[:BASIN_LEVEL]
        sub = gdf[gdf["l5"].isin(present)].dissolve(by="l5").reset_index()
        sub["split"] = sub["l5"].map(basin_split)
        sub["color"] = sub["split"].map(SPLIT_COLORS)

        world = gpd.read_file(shpreader.natural_earth(
            resolution="110m", category="cultural", name="admin_0_countries"))

        fig, ax = plt.subplots(figsize=(16, 7))
        world.plot(ax=ax, color="#e8e8e8", edgecolor="#b0b0b0", linewidth=0.3, zorder=1)
        sub.plot(ax=ax, color=sub["color"], edgecolor="#404040", linewidth=0.3, zorder=2)
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 85); ax.set_axis_off()
        ax.set_title("Train/validation/test split by HydroBASINS Level-5 basin", pad=6)
        handles = [mpatches.Patch(color=c, label=s) for s, c in SPLIT_COLORS.items()]
        ax.legend(handles=handles, loc="lower left", framealpha=0.9)
        plt.tight_layout(pad=0.5)
        fig.savefig(SPLIT_MAP_PNG, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved {SPLIT_MAP_PNG}")
    except Exception as e:
        print(f"  ! Split map skipped ({type(e).__name__}: {e})")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  Script 5: Create Train/Val/Test Splits (activation + basin exclusive, L5)")
    print("=" * 80)

    config.migrate_csv_names()  # rename any old-named metadata files in place

    if not DATASET_METADATA_CSV.exists():
        print(f"ERROR: {DATASET_METADATA_CSV} not found. Run Script 4 first.")
        sys.exit(1)

    with open(DATASET_METADATA_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    print(f"\nLoaded {len(rows)} activations")

    # Per-event basin(s) and activation
    event_basins     = {}
    event_activation = {}
    event_res        = {}
    unmapped = []
    for r in rows:
        fn = r["folder_name"]
        basins = basin_l5_for_event(r.get("basin_id", ""))
        if not basins:
            unmapped.append(fn)
        event_basins[fn]     = basins
        event_activation[fn] = fn.split("_")[0]
        event_res[fn]        = (r.get("resolution_class") or "").strip() or "medium"
    if unmapped:
        print(f"  ! {len(unmapped)} events had no Level-5 basin (kept, activation-only grouping)")

    # Union-find over basins and activations; events join their activation + basins
    uf = UnionFind()
    for fn in event_basins:
        anchor = ("activation", event_activation[fn])
        uf.union(("event", fn), anchor)
        for b in event_basins[fn]:
            uf.union(anchor, ("basin", b))

    comp_events = defaultdict(list)
    for r in rows:
        fn = r["folder_name"]
        comp_events[uf.find(("event", fn))].append(r)

    components = []
    for key, evs in comp_events.items():
        res_counts = defaultdict(int)
        for r in evs:
            res_counts[event_res[r["folder_name"]]] += 1
        components.append({"key": str(key), "events": evs, "res_counts": res_counts})
    print(f"  {len(components)} connected components (groups that must stay together)")

    totals = {"n": len(rows)}
    for rcl in RES_CLASSES:
        totals[rcl] = sum(1 for r in rows if event_res[r["folder_name"]] == rcl)

    assigned = assign_components(components, totals)
    event_split = {}
    for s, evs in assigned.items():
        for r in evs:
            event_split[r["folder_name"]] = s

    # ── Write split column back ──────────────────────────────────────────────
    for r in rows:
        r["split"] = event_split.get(r["folder_name"], "train")
    out_fields = [c for c in fieldnames if c != "split"] + ["split"]
    with open(DATASET_METADATA_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n  ✓ Wrote split column to {DATASET_METADATA_CSV.name}")

    # ── Verify exclusivity ───────────────────────────────────────────────────
    basin_to_splits      = defaultdict(set)
    activation_to_splits = defaultdict(set)
    for fn, s in event_split.items():
        activation_to_splits[event_activation[fn]].add(s)
        for b in event_basins[fn]:
            basin_to_splits[b].add(s)
    basin_leaks      = sum(1 for v in basin_to_splits.values() if len(v) > 1)
    activation_leaks = sum(1 for v in activation_to_splits.values() if len(v) > 1)

    # ── Report + split_info.json ─────────────────────────────────────────────
    print("\n  Split        events   basins   medium  high  very-high")
    info = {"split_method": "activation_basin_exclusive_connected_components",
            "basin_level": BASIN_LEVEL,
            "target_ratios": TARGET_RATIOS,
            "n_events": len(rows),
            "n_components": len(components),
            "event_counts": {}, "event_ratios": {}, "basin_counts": {},
            "resolution_counts": {},
            "basin_overlap": basin_leaks, "activation_overlap": activation_leaks}
    for s in ["train", "val", "test"]:
        evs = assigned[s]
        n = len(evs)
        basins = set().union(*[event_basins[r["folder_name"]] for r in evs]) if evs else set()
        rc = {rcl: sum(1 for r in evs if event_res[r["folder_name"]] == rcl) for rcl in RES_CLASSES}
        info["event_counts"][s] = n
        info["event_ratios"][s] = round(n / max(len(rows), 1), 3)
        info["basin_counts"][s] = len(basins)
        info["resolution_counts"][s] = rc
        print(f"  {s:11s}{n:7d}{len(basins):9d}{rc['medium']:9d}{rc['high']:6d}{rc['very-high']:11d}")
    print(f"\n  basin overlap: {basin_leaks}   activation overlap: {activation_leaks}")

    with open(SPLIT_INFO_JSON, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  ✓ Wrote {SPLIT_INFO_JSON.name}")

    # ── Map ──────────────────────────────────────────────────────────────────
    print("\nDrawing basin split map ...")
    make_split_map(event_basins, event_split)

    print("\n" + "=" * 80)
    print(f"DONE  train={info['event_counts']['train']} "
          f"val={info['event_counts']['val']} test={info['event_counts']['test']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
