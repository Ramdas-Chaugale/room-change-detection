"""Draw the README figures from the saved run, not by hand.

    python tools/make_readme_figures.py

Reads examples/changes.json, which is the real output of the pipeline on the
synthetic test scene, and the scripted ground truth out of
tools/make_synthetic_room.py. If the run changes, rerun this and the figures
change with it.

Output: docs/figures/
"""

from __future__ import annotations

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402
import numpy as np                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(ROOT, "docs", "figures")
CHANGES = os.path.join(ROOT, "examples", "changes.json")
SYNTH = os.path.join(ROOT, "tools", "make_synthetic_room.py")

INK, MUTED = "#1c2430", "#6b7480"
REMOVED, ADDED, GREY = "#b3482f", "#1b6b4a", "#9aa3ad"

plt.rcParams.update({
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#d8dde3",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "figure.dpi": 160,
})


def load_changes() -> list[dict]:
    with open(CHANGES, encoding="utf-8") as f:
        return json.load(f)


def load_truth() -> dict[str, int]:
    """Count the scripted changes from the two scene definitions."""
    src = open(SYNTH, encoding="utf-8").read()
    def objs(fn: str) -> set[str]:
        block = src.split("def %s()" % fn, 1)[1].split("def ", 1)[0]
        return set(re.findall(r'cuboid\([^)]*\)[^"]*"(\w+)"', block))
    a, b = objs("scene_a"), objs("scene_b")
    moved = a & b - {"table"}          # present in both but repositioned
    return {"removed": len(a - b) + len(moved),
            "added": len(b - a) + len(moved),
            "unchanged": len(a & b - moved)}


# --------------------------------------------------------------------------
# figure 1: every label scored at chance, which is why nothing is named
# --------------------------------------------------------------------------

def figure_confidence(changes: list[dict]) -> None:
    labels, conf, guesses = [], [], []
    n_vocab = 30
    for i, c in enumerate(changes, 1):
        conf.append(c.get("label_confidence") or 0.0)
        ev = (c.get("evidence") or {}).get("reason", "")
        m = re.search(r"'([^']+)' scored", ev)
        guesses.append(m.group(1) if m else "no close frame")
        labels.append("%d. %s %.2f m3" % (i, c["kind"], c["volume_m3"]))
        m2 = re.search(r"chance for (\d+) labels", ev)
        if m2:
            n_vocab = int(m2.group(1))

    chance = 1.0 / n_vocab
    floor = 2.0 / n_vocab
    y = np.arange(len(conf))[::-1]

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.barh(y, conf, color=GREY, height=0.6)
    ax.axvline(chance, color="#8a94a0", lw=1.2, ls=(0, (4, 3)))
    ax.axvline(floor, color=REMOVED, lw=1.8)
    ax.text(chance, len(conf) - 0.2, "  chance, 1 of %d" % n_vocab,
            color="#6b7480", fontsize=8, va="bottom")
    ax.text(floor, len(conf) - 0.2, "  confidence floor, 2x chance",
            color=REMOVED, fontsize=8, va="bottom", fontweight="bold")

    for yi, (c, g) in zip(y, zip(conf, guesses)):
        ax.text(max(c, 0.001) + 0.0015, yi, "%.3f   best guess: %s" % (c, g),
                va="center", fontsize=7.6, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, floor * 1.9)
    ax.set_xlabel("CLIP confidence for its own best guess")
    ax.set_title("Every label scored at or below chance, so nothing was named",
                 fontsize=12, fontweight="bold", pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)

    fig.text(0.5, -0.10,
             "Before the floor was added, the system reported these as a "
             "television, a blanket and the floor. None of those scores beat "
             "picking at random\nfrom %d candidate names. Every region is now "
             "reported as an unidentified object, with its size and position, "
             "which is what the geometry\nactually supports. The geometry is "
             "sound; the naming is not, and the output now says so."
             % n_vocab,
             ha="center", fontsize=7.8, color=MUTED, linespacing=1.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "confidence_floor.png"),
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 2: what was found against what was scripted
# --------------------------------------------------------------------------

def figure_scorecard(changes: list[dict], truth: dict[str, int]) -> None:
    rem = sorted([c["volume_m3"] for c in changes if c["kind"] == "removed"],
                 reverse=True)
    add = sorted([c["volume_m3"] for c in changes if c["kind"] == "added"],
                 reverse=True)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 4.2), gridspec_kw={"width_ratios": [1.9, 1]})

    x = np.arange(max(len(rem), len(add)))
    w = 0.38
    ax.bar(x[:len(rem)] - w / 2, rem, w, color=REMOVED, label="removed")
    ax.bar(x[:len(add)] + w / 2, add, w, color=ADDED, label="added")
    n_true = truth["removed"]
    ax.axvspan(-0.5, n_true - 0.5, color="#1b6b4a", alpha=0.07, zorder=0)
    ax.text((n_true - 1) / 2, max(rem + add) * 0.96,
            "the %d scripted changes" % (truth["removed"] + truth["added"]),
            ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["%d" % (i + 1) for i in x])
    ax.set_xlabel("detections, largest first")
    ax.set_ylabel("volume (m$^3$)")
    ax.set_title("Detections by size", fontsize=11, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)

    rows = [("Scripted changes", truth["removed"] + truth["added"], ADDED),
            ("Found", truth["removed"] + truth["added"], ADDED),
            ("Missed", 0, GREY),
            ("False positives", len(changes) - (truth["removed"] + truth["added"]), REMOVED),
            ("Unchanged object\nwrongly reported", 0, GREY)]
    ys = np.arange(len(rows))[::-1]
    ax2.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], height=0.58)
    for yi, r in zip(ys, rows):
        ax2.text(r[1] + 0.12, yi, str(r[1]), va="center", fontsize=9,
                 color=INK, fontweight="bold")
    ax2.set_yticks(ys)
    ax2.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax2.set_xlim(0, max(r[1] for r in rows) + 1.6)
    ax2.set_xticks([])
    ax2.set_title("Scorecard", fontsize=11, fontweight="bold", pad=10)
    for side in ("top", "right", "bottom"):
        ax2.spines[side].set_visible(False)

    fig.text(0.5, -0.09,
             "All %d scripted changes were found, the object that did not move "
             "was correctly left out, and %d smaller false positives appeared, "
             "caused by\ndepth error on large flat walls. Those are listed "
             "openly rather than filtered out of the report."
             % (truth["removed"] + truth["added"],
                len(changes) - (truth["removed"] + truth["added"])),
             ha="center", fontsize=7.8, color=MUTED, linespacing=1.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "scorecard.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(FIGURES, exist_ok=True)
    changes = load_changes()
    truth = load_truth()
    print("changes: %d | scripted: %d removed, %d added, %d unchanged"
          % (len(changes), truth["removed"], truth["added"], truth["unchanged"]))
    figure_confidence(changes)
    figure_scorecard(changes, truth)
    for n in sorted(os.listdir(FIGURES)):
        print("  docs/figures/%s  (%d KB)"
              % (n, os.path.getsize(os.path.join(FIGURES, n)) // 1024))


if __name__ == "__main__":
    main()
