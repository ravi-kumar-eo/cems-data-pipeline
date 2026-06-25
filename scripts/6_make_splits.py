#!/usr/bin/env python3
"""
Script 6: Make train/val/test split

Assigns every patch to a train, validation, or test split and writes the
assignment into the `split` column of both the patch index and the event
catalog, plus three split index files (train/val/test_patches.csv) and a set of
balance plots.

The split runs after patch extraction (Step 5) because it balances by PATCH
count: the weight of each event is how many patches it yields, which is only
known once the patches exist.

Two HARD constraints are always enforced, so every run produces a leak-free
split:
  1. Level-5 basin exclusivity  - no HydroBASINS Pfafstetter Level-5 basin
     appears in more than one split.
  2. Whole-event integrity       - every event lives entirely in one split; no
     event is fragmented.
Events that share a Level-5 basin must therefore travel together, so the script
builds the connected components of the event-to-basin graph (union-find) and
assigns whole components.

Everything else is a SOFT objective, applied best-effort so the result is
balanced but a valid split is always returned:
  patch share ~70/15/15  ->  per-continent  ->  resolution class  ->  event size.
Components are assigned per continent, largest first, to the split with the
biggest relative patch deficit; a component larger than a continent's val/test
target is forced into train. Climate is intentionally not balanced: a few large,
climate-homogeneous components dominate the patch count and cannot be divided, so
the climate mix is fixed by how those components fall.

Input
  data/metadata/released_patches_metadata.csv   one row per patch (from Step 5)
  data/metadata/released_events_metadata.csv     one row per event (area_km2)

Output
  data/metadata/released_patches_metadata.csv    `split` column written/updated
  data/metadata/released_events_metadata.csv      `split` column written/updated
  data/metadata/split_global/{train,val,test}_patches.csv
  data/plots/splits/*.png                         balance plots (if matplotlib present)

Usage
  python scripts/6_make_splits.py
"""

import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("ERROR: numpy and pandas are required. Install with: pip install numpy pandas")
    sys.exit(1)

import config

PATCH_CSV   = config.CSV_PATCH_METADATA
EVENT_CSV   = config.CSV_COMPLETE_METADATA
SPLIT_DIR   = config.SPLIT_DIR
PLOTS_DIR   = config.PLOTS_DIR

# Soft target by patch share. The achieved ratios drift from this because the
# largest indivisible components are forced into train.
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS = ["train", "val", "test"]


# ── EVENT ATTRIBUTES (aggregated from the patches) ───────────────────────────
def event_size_bin(area_km2):
    a = float(area_km2) if pd.notna(area_km2) else 0.0
    if a < 100:
        return "small<100"
    if a < 1000:
        return "med100-1k"
    return "large>1k"


def basin_codes(value):
    """Level-5 codes for a patch: a dash-joined string of one or more codes."""
    return set(b for b in str(value).split("-") if b and b != "nan")


# ── UNION-FIND OVER EVENTS SHARING AN L5 BASIN ───────────────────────────────
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


def build_components(df):
    """Group events into connected components linked by shared L5 basins."""
    events = df["folder_name"].unique()
    uf = UnionFind()
    for fn in events:
        uf.find(fn)
    basin_to_events = defaultdict(list)
    ev_basins = df.groupby("folder_name")["basin_id"].agg(
        lambda s: set().union(*(basin_codes(v) for v in s)))
    for fn in events:
        for b in ev_basins[fn]:
            basin_to_events[b].append(fn)
    for evs in basin_to_events.values():
        for other in evs[1:]:
            uf.union(evs[0], other)
    comp_of = {fn: uf.find(fn) for fn in events}
    comps = defaultdict(list)
    for fn, c in comp_of.items():
        comps[c].append(fn)
    return comp_of, comps


# ── ASSIGNMENT ───────────────────────────────────────────────────────────────
def assign_splits(df, ev):
    # Continent drives the soft per-continent balancing. If the catalog has no
    # continent column (an older Step 4), fall back to one global bucket: the
    # hard basin/event constraints still hold and a valid split is still made.
    if "continent" not in df.columns or df["continent"].isna().all():
        df = df.copy()
        df["continent"] = "all"
    e_npatch = df.groupby("folder_name").size()
    mode = lambda s: s.mode().iat[0]
    e_cont = df.groupby("folder_name")["continent"].agg(mode)
    e_res = df.groupby("folder_name")["resolution_class"].agg(mode)
    area = ev.set_index("folder_name")["area_km2"].to_dict()
    e_size = {fn: event_size_bin(area.get(fn, np.nan)) for fn in e_npatch.index}

    comp_of, comps = build_components(df)
    biggest = max(len(v) for v in comps.values())
    print(f"  events={len(e_npatch)}  components={len(comps)}  "
          f"largest component={biggest} events")

    # per-component patch totals broken down by the soft strata
    def strata(fn):
        n = int(e_npatch[fn])
        return {("cont", e_cont[fn]): n, ("res", e_res[fn]): n, ("size", e_size[fn]): n}

    comp_list = []
    for c, fns in comps.items():
        st = defaultdict(int)
        for f in fns:
            for k, v in strata(f).items():
                st[k] += v
        comp_list.append({"comp": c, "fns": fns,
                          "n": int(sum(e_npatch[f] for f in fns)), "st": st})
    # global largest-first order fixes the per-continent assignment sequence and
    # the shared resolution/size tie-break, so the split is deterministic.
    comp_list.sort(key=lambda x: -x["n"])
    for cl in comp_list:
        conts = {k[1]: v for k, v in cl["st"].items() if k[0] == "cont"}
        cl["cont"] = max(conts, key=conts.get)

    # global resolution/size totals, for a small secondary tie-break
    rs_total = defaultdict(int)
    for cl in comp_list:
        for k, v in cl["st"].items():
            if k[0] in ("res", "size"):
                rs_total[k] += v
    rs_ach = defaultdict(lambda: defaultdict(float))

    def rs_score(cl, s):
        sc = 0.0
        for k, v in cl["st"].items():
            if k[0] in ("res", "size") and rs_total[k] > 0:
                want = rs_total[k] * TARGET[s]
                sc += max(0.0, (want - rs_ach[s][k]) / want) * (v / cl["n"])
        return sc

    by_cont = defaultdict(list)
    for cl in comp_list:
        by_cont[cl["cont"]].append(cl)

    assign = {}
    for cont, cls in by_cont.items():
        cls.sort(key=lambda x: -x["n"])
        tot = sum(c["n"] for c in cls)
        target = {s: TARGET[s] * tot for s in SPLITS}
        achieved = {s: 0 for s in SPLITS}
        cap = max(target["val"], target["test"])
        for cl in cls:
            if cl["n"] > cap:
                best = "train"  # oversized component -> train (keeps val/test small)
            else:
                best = max(SPLITS, key=lambda s: (
                    (target[s] - achieved[s]) / max(target[s], 1e-9)
                    + 0.25 * rs_score(cl, s), s == "train"))
            assign[cl["comp"]] = best
            achieved[best] += cl["n"]
            for k, v in cl["st"].items():
                if k[0] in ("res", "size"):
                    rs_ach[best][k] += v

    return {fn: assign[comp_of[fn]] for fn in e_npatch.index}


# ── LEAKAGE CHECK ────────────────────────────────────────────────────────────
def check_leakage(df):
    exb = df[["basin_id", "split"]].copy()
    exb["basin_id"] = exb["basin_id"].astype(str).str.split("-")
    exb = exb.explode("basin_id")
    exb = exb[(exb["basin_id"] != "") & (exb["basin_id"] != "nan")]
    basin_span = exb.groupby("basin_id")["split"].nunique()
    event_span = df.groupby("folder_name")["split"].nunique()
    nb = int((basin_span > 1).sum())
    ne = int((event_span > 1).sum())
    print(f"  L5 basins spanning >1 split: {nb} of {basin_span.size}")
    print(f"  events spanning >1 split:    {ne} of {event_span.size}")
    return nb == 0 and ne == 0


# ── PLOTS (optional) ─────────────────────────────────────────────────────────
def make_plots(df, ev):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed - skipping balance plots")
        return
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    area = ev.set_index("folder_name")["area_km2"].to_dict()

    def sz(fn):
        a = float(area.get(fn, 0)) if pd.notna(area.get(fn, 0)) else 0
        return "small\n(<100)" if a < 100 else ("medium\n(100-1k)" if a < 1000 else "large\n(>1k)")

    df = df.copy()
    df["event_size"] = df["folder_name"].map(sz)
    col = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    tot = len(df)

    def stacked(column, order, title, fname):
        order = [o for o in order if o in set(df[column])]
        if not order:
            return
        ct = pd.crosstab(df[column], df["split"]).reindex(order).fillna(0)
        n = ct.sum(1)
        frac = ct.div(n, axis=0) * 100
        fig, ax = plt.subplots(figsize=(11, 0.9 * len(order) + 1.8))
        y = np.arange(len(order))[::-1]
        left = np.zeros(len(order))
        for s in SPLITS:
            vals = frac.get(s, pd.Series(0, index=order)).values
            ax.barh(y, vals, left=left, color=col[s], label=s, edgecolor="white", height=0.62)
            for yi, (v, l) in enumerate(zip(vals, left)):
                if v >= 5:
                    ax.text(l + v / 2, y[yi], f"{v:.0f}", va="center", ha="center",
                            color="white", fontsize=9, fontweight="bold")
            left += vals
        for gx in (70, 85):
            ax.axvline(gx, ls="--", c="#555", lw=0.8, zorder=5)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=10)
        for yi, cat in enumerate(order):
            ax.text(101.5, y[yi], f"{int(n[cat]):,}  ({n[cat] / tot * 100:.1f}% of all)",
                    va="center", ha="left", fontsize=9, color="#222")
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of this category's patches -> split (target: train 70 | val 15 | test 15)")
        ax.set_title(title, fontsize=13, pad=8)
        ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
        fig.subplots_adjust(right=0.74)
        fig.savefig(PLOTS_DIR / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")

    stacked("continent", df["continent"].value_counts().index.tolist(),
            "Continent balance - split distribution within each continent", "5_continent.png")
    stacked("climate", ["C Temperate", "D Cold", "B Arid", "A Tropical", "E Polar"],
            "Climate balance (Koppen) - split distribution within each climate", "2_climate.png")
    stacked("resolution_class", ["medium", "high", "very-high"],
            "Resolution-class balance - split distribution within each class", "4_resolution_class.png")
    stacked("event_size", ["large\n(>1k)", "medium\n(100-1k)", "small\n(<100)"],
            "Event-size balance (patch-weighted) - split distribution within each bin", "3_event_size.png")

    # overall share, patches vs events
    pr = df["split"].value_counts() / tot * 100
    ec = df.groupby("split")["folder_name"].nunique()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    ax.bar(x - 0.2, [pr.get(s, 0) for s in SPLITS], 0.4, label="patches",
           color=[col[s] for s in SPLITS])
    ax.bar(x + 0.2, [ec.get(s, 0) / ec.sum() * 100 for s in SPLITS], 0.4, label="events",
           alpha=0.45, color=[col[s] for s in SPLITS])
    for i, s in enumerate(SPLITS):
        ax.text(i - 0.2, pr.get(s, 0) + 1, f"{pr.get(s, 0):.1f}%", ha="center", fontsize=9)
        ax.text(i + 0.2, ec.get(s, 0) / ec.sum() * 100 + 1, f"{ec.get(s, 0)}e",
                ha="center", fontsize=9)
    for gy in (70, 15):
        ax.axhline(gy, ls="--", c="#999", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{int(pr.get(s, 0) / 100 * tot):,}p" for s in SPLITS])
    ax.set_ylabel("% of total")
    ax.set_title("Overall share (patches vs events)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "1_overall_share.png", dpi=130)
    plt.close(fig)
    print("  saved 1_overall_share.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Script 6: Make train/val/test split")
    print("=" * 70)

    if not PATCH_CSV.exists():
        print(f"ERROR: patch index not found: {PATCH_CSV}\n"
              f"Run Step 5 (5_make_patches.py) first.")
        sys.exit(1)

    df = pd.read_csv(PATCH_CSV)
    ev = pd.read_csv(EVENT_CSV) if EVENT_CSV.exists() else pd.DataFrame(
        columns=["folder_name", "area_km2"])
    print(f"  patches={len(df):,}  events(catalog)={len(ev):,}")

    ev_split = assign_splits(df, ev)
    df["split"] = df["folder_name"].map(ev_split)

    if not check_leakage(df):
        print("ERROR: leakage detected - aborting without writing.")
        sys.exit(1)

    ratios = (df["split"].value_counts() / len(df) * 100).round(1).to_dict()
    counts = df.groupby("split")["folder_name"].nunique().to_dict()
    print(f"  patch ratios: " +
          "  ".join(f"{s} {ratios.get(s, 0)}%" for s in SPLITS))
    print(f"  event counts: " +
          "  ".join(f"{s} {counts.get(s, 0)}" for s in SPLITS))

    # write split into both metadata files
    df.to_csv(PATCH_CSV, index=False)
    if not ev.empty:
        ev["split"] = ev["folder_name"].map(ev_split).fillna(ev.get("split"))
        ev.to_csv(EVENT_CSV, index=False)

    # three split index files
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for s, path in zip(SPLITS, (config.CSV_TRAIN_PATCHES, config.CSV_VAL_PATCHES,
                                config.CSV_TEST_PATCHES)):
        df[df["split"] == s].to_csv(path, index=False)
    print(f"  wrote split column + {SPLIT_DIR.name}/{{train,val,test}}_patches.csv")

    make_plots(df, ev)
    print("  done.")


if __name__ == "__main__":
    main()
