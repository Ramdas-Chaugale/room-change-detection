"""Say what happened, in a browser and in English.

Two outputs. An interactive 3D view where the unchanged room is grey and the
changes are coloured, so the geometry can be inspected rather than taken on
trust, and a written summary that reads like a person describing the room.

Positions are given relative to the room rather than in raw coordinates, since
"near the left wall" is useful to a reader and "x = 1.84" is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .changes import Cluster
from .occupancy import SessionMap, unpack

COLOURS = {
    "removed": "#d62728",
    "added": "#2ca02c",
    "moved": "#ff7f0e",
}


def _describe_position(centroid: np.ndarray, bounds_min: np.ndarray,
                       bounds_max: np.ndarray) -> str:
    """Rough plain-English placement inside the reconstructed volume."""
    span = np.maximum(bounds_max - bounds_min, 1e-6)
    rel = (centroid - bounds_min) / span

    parts = []
    if rel[0] < 0.33:
        parts.append("the left side")
    elif rel[0] > 0.66:
        parts.append("the right side")
    else:
        parts.append("the middle")

    if rel[2] < 0.33:
        depth = "near the front"
    elif rel[2] > 0.66:
        depth = "towards the back"
    else:
        depth = "in the centre"

    height = ""
    if rel[1] > 0.7:
        height = ", low down"
    elif rel[1] < 0.3:
        height = ", up high"

    return f"{depth} on {parts[0]}{height}"


def write_scene(path: Path, map_a: SessionMap, map_b: SessionMap,
                clusters: list[Cluster], voxel_size: float) -> None:
    """Interactive 3D HTML of the room with the changes highlighted."""
    fig = go.Figure()

    # The unchanged room, thinned out so the file stays openable.
    stable = np.intersect1d(map_a.occupied, map_b.occupied, assume_unique=True)
    if stable.size:
        pts = (unpack(stable).astype(np.float64) + 0.5) * voxel_size
        if len(pts) > 40000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 40000, replace=False)]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 2], z=-pts[:, 1],
            mode="markers",
            marker=dict(size=1.2, color="#b0b0b0", opacity=0.35),
            name=f"unchanged ({len(stable):,} voxels)",
            hoverinfo="skip",
        ))

    for index, cluster in enumerate(clusters):
        pts = (cluster.voxels.astype(np.float64) + 0.5) * voxel_size
        role = cluster.evidence.get("role", "")
        suffix = f" ({role})" if role else ""
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 2], z=-pts[:, 1],
            mode="markers",
            marker=dict(size=3.0, color=COLOURS[cluster.kind], opacity=0.9),
            name=f"{cluster.kind}: {cluster.label}{suffix}",
            text=[f"{cluster.label}<br>{cluster.volume_m3:.3f} m3"] * len(pts),
            hoverinfo="text",
        ))

    fig.update_layout(
        title="What changed in the room",
        scene=dict(
            xaxis_title="x", yaxis_title="depth", zaxis_title="height",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(path), include_plotlyjs="cdn")


def write_report(path: Path, clusters: list[Cluster], map_a: SessionMap,
                 map_b: SessionMap, voxel_size: float) -> str:
    """Natural language summary. Returns the text as well as writing it."""
    stable = np.intersect1d(map_a.occupied, map_b.occupied, assume_unique=True)
    everything = np.concatenate([
        (unpack(stable).astype(np.float64) + 0.5) * voxel_size
    ]) if stable.size else np.zeros((1, 3))
    bounds_min, bounds_max = everything.min(axis=0), everything.max(axis=0)

    lines: list[str] = ["# What changed in the room", ""]

    if not clusters:
        lines += ["No changes were detected between the two recordings.", "",
                  "Either nothing moved, or the two sessions did not observe enough "
                  "of the same space for a comparison to be possible. The overlap "
                  "figure in the run log tells you which."]
        text = "\n".join(lines)
        path.write_text(text, encoding="utf-8")
        return text

    seen_moves: set[int] = set()
    described: list[str] = []

    for index, cluster in enumerate(clusters):
        place = _describe_position(cluster.centroid, bounds_min, bounds_max)
        size = f"{cluster.volume_m3:.2f} m3"

        if cluster.kind == "moved":
            if index in seen_moves or cluster.partner is None:
                continue
            seen_moves.add(index)
            seen_moves.add(cluster.partner)
            other = clusters[cluster.partner]
            source = cluster if cluster.evidence.get("role") == "from" else other
            target = other if source is cluster else cluster
            distance = cluster.evidence.get("distance_m", 0.0)
            described.append(
                f"- **{cluster.label} moved.** It was {_describe_position(source.centroid, bounds_min, bounds_max)} "
                f"and is now {_describe_position(target.centroid, bounds_min, bounds_max)}, "
                f"about {distance:.2f} m away. Roughly {size}."
            )
        elif cluster.kind == "removed":
            described.append(
                f"- **{cluster.label} was removed** from {place}. "
                f"It occupied about {size}, and that space is now observed to be empty."
            )
        else:
            described.append(
                f"- **{cluster.label} was added** at {place}. "
                f"It occupies about {size}, in space that was observed to be empty before."
            )

    lines += [f"{len(described)} change(s) detected.", ""]
    lines += described
    lines += ["", "## How to read this", "",
              "A change is only reported where one session saw an object and the "
              "other session positively saw through that same space. Anywhere only "
              "one recording looked is left out, so objects hidden behind something "
              "in one pass are not reported as having vanished.", ""]

    confident = [c for c in clusters if c.confidence >= 0.5]
    if confident:
        lines.append(f"Labels come from CLIP over image crops. "
                     f"{len(confident)} of {len(clusters)} clusters were labelled with "
                     f"confidence above 0.5, the rest are the model's best guess.")

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def write_json(path: Path, clusters: list[Cluster]) -> None:
    """Machine readable version of the same findings."""
    payload = []
    for cluster in clusters:
        payload.append({
            "kind": cluster.kind,
            "label": cluster.label,
            "label_confidence": round(cluster.confidence, 3),
            "centroid": [round(float(v), 4) for v in cluster.centroid],
            "volume_m3": round(cluster.volume_m3, 4),
            "extent_m": [round(float(v), 4) for v in cluster.extent_m],
            "n_voxels": cluster.n_voxels,
            "evidence": {k: (v if not isinstance(v, (tuple, list)) else list(v))
                         for k, v in cluster.evidence.items()},
        })
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
