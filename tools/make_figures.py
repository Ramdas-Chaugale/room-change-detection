"""Draw the figures that go in the README.

Two of them come from a real run rather than being illustrations, so they
cannot drift away from what the code actually does:

  topdown.png        the room from above, unchanged structure in grey and the
                     detected changes in colour, which is the result in one look
  change_summary.png every cluster by volume and kind, so the signal and the
                     noise are visible side by side

The third is a diagram of the three-state occupancy logic, because that idea is
the whole project and a paragraph does not land the way a picture does.

    python tools/make_figures.py --run output/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

COLOURS = {"removed": "#d62728", "added": "#2ca02c", "moved": "#ff7f0e"}
GREY = "#9a9a9a"


def _style(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def change_summary(clusters: list[dict], out: Path) -> None:
    """Every detected cluster by volume, grouped by kind."""
    if not clusters:
        print("  no clusters, skipping change_summary")
        return

    order = sorted(clusters, key=lambda c: -c["volume_m3"])
    labels = []
    for i, c in enumerate(order):
        name = c["label"]
        if name in ("unlabelled", "unidentified object"):
            name = "unidentified"
        labels.append(f"{name}\n{c['kind']}")
    volumes = [c["volume_m3"] for c in order]
    colours = [COLOURS[c["kind"]] for c in order]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(range(len(order)), volumes, color=colours, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("cluster volume, cubic metres")
    ax.set_title("Detected changes, largest first", fontsize=12, loc="left")

    # The real objects sit well clear of the noise floor, so say where it is.
    if len(volumes) > 4:
        floor = float(np.median(volumes))
        ax.axhline(floor, color="#444444", linestyle="--", linewidth=0.9)
        ax.text(len(order) - 0.4, floor * 1.08, "median cluster", fontsize=7.5,
                ha="right", color="#444444")

    for bar, v in zip(bars, volumes):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7)

    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def topdown(clusters: list[dict], out: Path) -> None:
    """The room from above, with the changes in place.

    A plan view is the one projection where a reader can see at a glance that
    the detections sit in different parts of the room rather than piling up in
    one corner, which is what a plausible result looks like.
    """
    if not clusters:
        print("  no clusters, skipping topdown")
        return

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    # Circles sized by volume rather than bounding boxes. DBSCAN clusters
    # sprawl, so their boxes overlap into an unreadable mess, while a
    # centroid plus a size carries the same information cleanly.
    biggest = max(c["volume_m3"] for c in clusters)
    for c in clusters:
        x, _y, z = c["centroid"]
        kind = c["kind"]
        area = 120 + 2600 * (c["volume_m3"] / biggest)
        ax.scatter([x], [z], s=area, color=COLOURS[kind], alpha=0.45,
                   edgecolors=COLOURS[kind], linewidths=1.6, zorder=3)
        name = c["label"]
        if name in ("unlabelled", "unidentified object"):
            name = "?"
        # Only annotate the ones big enough to matter, so it stays legible.
        if c["volume_m3"] >= 0.05 * biggest:
            text = name + "\n" + format(c["volume_m3"], ".2f") + " m3"
            ax.annotate(text, (x, z), textcoords="offset points",
                        xytext=(0, -6), ha="center", va="top",
                        fontsize=7, color="#222222", zorder=4)

    # Pair up the two halves of any move so the displacement is visible.
    for i, c in enumerate(clusters):
        if c["kind"] != "moved":
            continue
        if c.get("evidence", {}).get("role") != "from":
            continue
        partners = [d for d in clusters
                    if d["kind"] == "moved" and d.get("evidence", {}).get("role") == "to"]
        if not partners:
            continue
        target = partners[0]
        ax.add_patch(FancyArrowPatch(
            (c["centroid"][0], c["centroid"][2]),
            (target["centroid"][0], target["centroid"][2]),
            arrowstyle="-|>", mutation_scale=16,
            color=COLOURS["moved"], linewidth=1.6, linestyle=":"))

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=10,
                          color=COLOURS[k], alpha=0.6, label=k)
               for k in ("removed", "added", "moved")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

    # add_patch does not autoscale, so the limits come from the data or the
    # whole plot renders empty.
    xs = [c["centroid"][0] for c in clusters]
    zs = [c["centroid"][2] for c in clusters]
    pad = 0.8
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(zs) - pad, max(zs) + pad)

    ax.set_xlabel("x, metres")
    ax.set_ylabel("depth, metres")
    ax.set_title("Where the changes are, seen from above", fontsize=12, loc="left")
    ax.set_aspect("equal")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def occupancy_states(out: Path) -> None:
    """Why free space is the thing that makes this work.

    Drawn rather than measured, because it explains a rule rather than
    reporting a result.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    titles = ["Session A", "Session B", "Comparison"]

    def room(ax):
        ax.add_patch(Rectangle((0, 0), 10, 6, facecolor="#f7f7f7",
                               edgecolor="#cccccc", linewidth=1.2))
        ax.plot(0.8, 0.9, marker="^", markersize=11, color="#333333")
        ax.text(0.8, 0.1, "camera", fontsize=7.5, ha="center", color="#333333")
        ax.set_xlim(-0.4, 10.4)
        ax.set_ylim(-0.6, 6.4)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # A: a chair present, and a region hidden behind a pillar
    ax = axes[0]
    room(ax)
    ax.add_patch(Rectangle((3.4, 2.6), 1.5, 1.5, facecolor=COLOURS["removed"],
                           alpha=0.5, edgecolor="none"))
    ax.text(4.15, 3.35, "chair", ha="center", va="center", fontsize=8)
    ax.add_patch(Rectangle((7.2, 1.0), 0.5, 3.4, facecolor="#555555", alpha=0.8))
    ax.text(7.45, 4.7, "pillar", ha="center", fontsize=7.5)
    ax.add_patch(Rectangle((7.9, 1.0), 2.0, 3.4, facecolor="#dddddd",
                           hatch="//", edgecolor="#aaaaaa", linewidth=0.5))
    ax.text(8.9, 2.7, "never seen", ha="center", va="center", fontsize=7.5, color="#666666")
    ax.set_title(titles[0], fontsize=10, loc="left")

    # B: chair gone, same region still hidden
    ax = axes[1]
    room(ax)
    ax.add_patch(Rectangle((3.4, 2.6), 1.5, 1.5, facecolor="none",
                           edgecolor="#bbbbbb", linestyle="--", linewidth=1.2))
    ax.text(4.15, 3.35, "empty,\nray passes\nthrough", ha="center", va="center", fontsize=7)
    ax.add_patch(Rectangle((7.2, 1.0), 0.5, 3.4, facecolor="#555555", alpha=0.8))
    ax.add_patch(Rectangle((7.9, 1.0), 2.0, 3.4, facecolor="#dddddd",
                           hatch="//", edgecolor="#aaaaaa", linewidth=0.5))
    ax.text(8.9, 2.7, "never seen", ha="center", va="center", fontsize=7.5, color="#666666")
    ax.set_title(titles[1], fontsize=10, loc="left")

    # Comparison: one reported, one withheld
    ax = axes[2]
    room(ax)
    ax.add_patch(Rectangle((3.4, 2.6), 1.5, 1.5, facecolor=COLOURS["removed"],
                           alpha=0.5, edgecolor=COLOURS["removed"], linewidth=1.6))
    ax.text(4.15, 3.35, "REMOVED", ha="center", va="center", fontsize=7.5,
            fontweight="bold", color="#7a1114")
    ax.text(4.15, 1.9, "occupied in A\nand free in B", ha="center", fontsize=7,
            color="#444444")
    ax.add_patch(Rectangle((7.2, 1.0), 0.5, 3.4, facecolor="#555555", alpha=0.8))
    ax.add_patch(Rectangle((7.9, 1.0), 2.0, 3.4, facecolor="#dddddd",
                           hatch="//", edgecolor="#aaaaaa", linewidth=0.5))
    ax.text(8.9, 2.9, "NO CLAIM", ha="center", va="center", fontsize=7.5,
            fontweight="bold", color="#555555")
    ax.text(8.9, 2.1, "unknown in\nboth", ha="center", fontsize=7, color="#666666")
    ax.set_title(titles[2], fontsize=10, loc="left")

    fig.suptitle("A gap only counts as a change when the other session saw through it",
                 fontsize=12, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=Path("output"), type=Path,
                        help="a finished run directory containing changes.json")
    parser.add_argument("--out", default=Path("docs/figures"), type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("drawing figures")
    occupancy_states(args.out / "occupancy_states.png")

    changes_file = args.run / "changes.json"
    if not changes_file.exists():
        print(f"  no {changes_file}, skipping the result figures. Run the pipeline first.")
        return 0

    clusters = json.loads(changes_file.read_text(encoding="utf-8"))
    change_summary(clusters, args.out / "change_summary.png")
    topdown(clusters, args.out / "topdown.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
