"""Turn posed depth maps into what each session knows about the room.

A point cloud on its own cannot answer "is the chair gone?". An empty patch in
session B means either the chair left or the camera never looked there, and
those are completely different claims. Treating them the same is how a change
detector ends up reporting half a room as deleted because somebody walked a
slightly different route the second time.

So each session gets a three-state map rather than a cloud:

  OCCUPIED  depth measurements landed in this voxel
  FREE      a camera ray passed through this voxel on its way to a surface,
            so the space was looked at and there was nothing in it
  UNKNOWN   neither, which means no claim can be made

FREE is the expensive one and the reason the rest works. Carving it means
marching every sampled ray from the camera centre up to just short of where it
hit something, and marking what it passed through as empty.

Voxel coordinates are packed into a single int64 so the set algebra in
changes.py is numpy rather than Python loops.
"""

from __future__ import annotations

import numpy as np

# 21 bits per axis, signed, which covers plus or minus a million voxels.
_BITS = 21
_MASK = (1 << _BITS) - 1
_HALF = 1 << (_BITS - 1)


def pack(voxels: np.ndarray) -> np.ndarray:
    """(N,3) integer voxel coordinates into (N,) int64 keys."""
    v = voxels.astype(np.int64) + _HALF
    if np.any(v < 0) or np.any(v > _MASK):
        raise ValueError("scene too large for the voxel packing, raise voxel_size")
    return (v[:, 0] << (2 * _BITS)) | (v[:, 1] << _BITS) | v[:, 2]


def unpack(keys: np.ndarray) -> np.ndarray:
    """int64 keys back into (N,3) integer voxel coordinates."""
    x = ((keys >> (2 * _BITS)) & _MASK) - _HALF
    y = ((keys >> _BITS) & _MASK) - _HALF
    z = (keys & _MASK) - _HALF
    return np.stack([x, y, z], axis=1)


class SessionMap:
    """Occupancy and free space accumulated over one session's frames."""

    def __init__(self, voxel_size: float, cfg: dict):
        self.voxel_size = float(voxel_size)
        self.min_hits = int(cfg["min_hits"])
        self.ray_stride = int(cfg["ray_stride"])
        self.surface_margin = float(cfg["surface_margin_m"])
        self._surface_keys: list[np.ndarray] = []
        self._free_keys: list[np.ndarray] = []
        self.occupied: np.ndarray = np.zeros(0, dtype=np.int64)
        self.free: np.ndarray = np.zeros(0, dtype=np.int64)
        # Kept so semantics.py can work out which frame best sees a cluster.
        self.frames: list[dict] = []

    def _voxelise(self, points: np.ndarray) -> np.ndarray:
        return np.floor(points / self.voxel_size).astype(np.int64)

    def add_frame(self, image_name: str, R_wc: np.ndarray, centre: np.ndarray,
                  directions: np.ndarray, depths: np.ndarray,
                  max_depth: float) -> None:
        """Fold one posed depth map into the map.

        directions are camera-space rays with z normalised to 1, depths are the
        matching distances along z. Rays with no valid depth still carve free
        space out to the sensible limit, because looking into the distance and
        seeing nothing is also information.
        """
        valid = np.isfinite(depths) & (depths > 0)
        if valid.sum() == 0:
            return

        dirs = directions[valid]
        d = depths[valid]

        # Surfaces.
        points_cam = dirs * d[:, None]
        points_world = points_cam @ R_wc.T + centre
        surface_voxels = self._voxelise(points_world)
        self._surface_keys.append(pack(surface_voxels))

        # Free space along each ray, in chunks so a big frame does not blow up.
        limit = np.maximum(d - self.surface_margin, 0.0)
        steps = int(np.ceil(min(max_depth, float(limit.max())) / self.voxel_size))
        if steps > 1:
            t = (np.arange(1, steps + 1, dtype=np.float64) * self.voxel_size)
            chunk = max(1, int(2_000_000 / max(steps, 1)))
            for start in range(0, len(dirs), chunk):
                sl = slice(start, start + chunk)
                along = t[None, :] < limit[sl, None]
                if not along.any():
                    continue
                pts_cam = dirs[sl][:, None, :] * t[None, :, None]
                pts = pts_cam[along]
                pts_world = pts @ R_wc.T + centre
                self._free_keys.append(pack(self._voxelise(pts_world)))

        self.frames.append({
            "name": image_name,
            "R_wc": R_wc,
            "centre": centre,
        })

    def finalise(self) -> None:
        """Resolve the accumulated evidence into occupied and free sets."""
        if self._surface_keys:
            keys = np.concatenate(self._surface_keys)
            unique, counts = np.unique(keys, return_counts=True)
            self.occupied = unique[counts >= self.min_hits]
        self._surface_keys = []

        if self._free_keys:
            free = np.unique(np.concatenate(self._free_keys))
            # A voxel that anything ever landed in is a surface, not empty air.
            # Without this the front faces of objects carve themselves away.
            if self.occupied.size:
                free = free[~np.isin(free, self.occupied, assume_unique=True)]
            self.free = free
        self._free_keys = []

    @property
    def observed(self) -> np.ndarray:
        return np.union1d(self.occupied, self.free)

    def summary(self) -> str:
        return (f"{len(self.occupied):,} occupied voxels, "
                f"{len(self.free):,} free, "
                f"{len(self.frames)} frames fused")
