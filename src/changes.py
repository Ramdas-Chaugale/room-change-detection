"""Work out what actually changed between the two session maps.

The whole argument is in four lines of set algebra:

    removed = occupied in A  AND  free in B
    added   = occupied in B  AND  free A

Both sides require positive evidence from the other session. A voxel that was
occupied in A and simply never looked at in B produces nothing, because nothing
can be said about it. That is the difference between a change detector and a
viewpoint detector.

After that the changed voxels are clustered into objects, and a removal and an
addition that look like the same thing in two places are reported as one move
rather than two unrelated events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN

from .occupancy import SessionMap, unpack


@dataclass
class Cluster:
    """One connected lump of changed space."""
    kind: str                      # "removed", "added" or "moved"
    voxels: np.ndarray             # (N,3) integer voxel coordinates
    centroid: np.ndarray           # (3,) world coordinates
    volume_m3: float
    extent_m: np.ndarray           # (3,) bounding box size in world units
    label: str = "unlabelled"
    confidence: float = 0.0
    partner: int | None = None     # index of the matching cluster for a move
    evidence: dict = field(default_factory=dict)

    @property
    def n_voxels(self) -> int:
        return len(self.voxels)


def _cluster_keys(keys: np.ndarray, voxel_size: float, cfg: dict,
                  kind: str) -> list[Cluster]:
    """Group changed voxels into objects."""
    if keys.size == 0:
        return []

    voxels = unpack(keys)
    min_voxels = int(cfg["min_cluster_voxels"])
    eps = float(cfg["cluster_eps_voxels"])

    labels = DBSCAN(eps=eps, min_samples=4).fit_predict(voxels.astype(np.float64))

    clusters: list[Cluster] = []
    for label in np.unique(labels):
        if label < 0:
            continue
        member = voxels[labels == label]
        if len(member) < min_voxels:
            continue
        world = (member.astype(np.float64) + 0.5) * voxel_size
        clusters.append(Cluster(
            kind=kind,
            voxels=member,
            centroid=world.mean(axis=0),
            volume_m3=len(member) * voxel_size ** 3,
            extent_m=world.max(axis=0) - world.min(axis=0) + voxel_size,
        ))

    clusters.sort(key=lambda c: c.volume_m3, reverse=True)
    return clusters


def detect(map_a: SessionMap, map_b: SessionMap, voxel_size: float,
           cfg: dict) -> list[Cluster]:
    """Compare two session maps and return the change clusters."""
    removed_keys = np.intersect1d(map_a.occupied, map_b.free, assume_unique=True)
    added_keys = np.intersect1d(map_b.occupied, map_a.free, assume_unique=True)

    overlap = np.intersect1d(map_a.observed, map_b.observed, assume_unique=True)
    print(f"  shared observed volume: {len(overlap):,} voxels")
    print(f"  removed candidates: {len(removed_keys):,} voxels")
    print(f"  added candidates: {len(added_keys):,} voxels")
    if len(overlap) < 1000:
        print("  WARNING: the two sessions barely cover the same space. "
              "Anything found here is weak evidence.")

    clusters = (_cluster_keys(removed_keys, voxel_size, cfg, "removed")
                + _cluster_keys(added_keys, voxel_size, cfg, "added"))
    return clusters


def match_moves(clusters: list[Cluster], cfg: dict) -> list[Cluster]:
    """Pair a removal with an addition when it is the same object elsewhere.

    Matching is deliberately conservative. Two clusters have to agree on
    semantic label, be comparable in volume, and sit within a sensible distance
    before the pair is called a move. Anything that does not pair stays as a
    plain removal or addition, which is the honest answer when the evidence is
    only that something vanished.
    """
    max_distance = float(cfg["move_max_distance_m"])
    ratio_floor = float(cfg["move_volume_ratio"])

    removed = [i for i, c in enumerate(clusters) if c.kind == "removed"]
    added = [i for i, c in enumerate(clusters) if c.kind == "added"]

    pairs: list[tuple[float, int, int]] = []
    for i in removed:
        for j in added:
            a, b = clusters[i], clusters[j]
            if a.label != b.label:
                continue
            # Matching on a label the model was not confident about would pair
            # arbitrary objects that happen to share a guess.
            if a.label in ("unlabelled", "unidentified object"):
                continue
            ratio = min(a.volume_m3, b.volume_m3) / max(a.volume_m3, b.volume_m3)
            if ratio < ratio_floor:
                continue
            distance = float(np.linalg.norm(a.centroid - b.centroid))
            if distance > max_distance:
                continue
            # Prefer confident, similar, nearby pairs.
            cost = distance * (2.0 - ratio)
            pairs.append((cost, i, j))

    pairs.sort()
    taken: set[int] = set()
    for _cost, i, j in pairs:
        if i in taken or j in taken:
            continue
        taken.add(i)
        taken.add(j)
        clusters[i].kind = "moved"
        clusters[j].kind = "moved"
        clusters[i].partner = j
        clusters[j].partner = i
        clusters[i].evidence["role"] = "from"
        clusters[j].evidence["role"] = "to"
        clusters[i].evidence["distance_m"] = float(
            np.linalg.norm(clusters[i].centroid - clusters[j].centroid))
        clusters[j].evidence["distance_m"] = clusters[i].evidence["distance_m"]

    return clusters
