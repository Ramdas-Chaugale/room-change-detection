"""Render two videos of a synthetic room, with known changes between them.

This exists so the pipeline can be tested and so anyone can run the project
without filming anything first. Because the changes are scripted, the output can
be checked against the truth rather than eyeballed.

The renderer is deliberately simple: textured quads, projected with a pinhole
camera and composited back to front. Textures are blurred colour noise, which
gives SIFT plenty to match on. A clean untextured render would reconstruct
badly and that would be the renderer's fault, not the pipeline's.

    python tools/make_synthetic_room.py --out videos/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
FOCAL = 620.0
RNG = np.random.default_rng(7)


def texture(seed: int, size: int = 256, tint=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Blurred colour noise. Rich enough for feature matching."""
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(size // 8, size // 8, 3)).astype(np.uint8)
    img = cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    speckle = rng.integers(0, 40, size=(size, size, 1)).astype(np.uint8)
    img = cv2.add(img, np.repeat(speckle, 3, axis=2))
    img = (img.astype(np.float64) * np.array(tint)).clip(0, 255).astype(np.uint8)
    return img


class Quad:
    """A textured rectangle in world space."""

    def __init__(self, corners: np.ndarray, tex: np.ndarray, name: str):
        # corners: (4,3) in order top-left, top-right, bottom-right, bottom-left
        self.corners = corners.astype(np.float64)
        self.tex = tex
        self.name = name

    def centroid(self) -> np.ndarray:
        return self.corners.mean(axis=0)


def tile(corners: np.ndarray, tex: np.ndarray, name: str,
         nx: int = 6, ny: int = 6) -> list[Quad]:
    """Split a large quad into a grid of smaller ones.

    The near-plane test in render() drops any quad with a corner behind the
    camera. On a floor or a wall that spans the whole room this removes the
    entire surface as soon as the camera stands on it. Tiling keeps the test
    local, so only the tiles genuinely behind the camera vanish.
    """
    tl, tr, br, bl = corners
    th, tw = tex.shape[:2]
    out: list[Quad] = []
    for i in range(nx):
        for j in range(ny):
            u0, u1 = i / nx, (i + 1) / nx
            v0, v1 = j / ny, (j + 1) / ny

            def lerp(a, b, s):
                return a + (b - a) * s

            top0 = lerp(tl, tr, u0)
            top1 = lerp(tl, tr, u1)
            bot0 = lerp(bl, br, u0)
            bot1 = lerp(bl, br, u1)
            patch = np.array([
                lerp(top0, bot0, v0), lerp(top1, bot1, v0),
                lerp(top1, bot1, v1), lerp(top0, bot0, v1),
            ])
            crop = tex[int(v0 * th):max(int(v1 * th), int(v0 * th) + 2),
                       int(u0 * tw):max(int(u1 * tw), int(u0 * tw) + 2)]
            out.append(Quad(patch, np.ascontiguousarray(crop), f"{name}_{i}_{j}"))
    return out


def cuboid(centre, size, seed, name) -> list[Quad]:
    """Six faces of a box, each with its own texture."""
    cx, cy, cz = centre
    sx, sy, sz = np.asarray(size) / 2.0
    c = {
        "nxyz": np.array([cx - sx, cy - sy, cz - sz]),
        "pxyz": np.array([cx + sx, cy + sy, cz + sz]),
    }
    x0, y0, z0 = c["nxyz"]
    x1, y1, z1 = c["pxyz"]

    faces = {
        "front":  [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
        "back":   [[x1, y0, z1], [x0, y0, z1], [x0, y1, z1], [x1, y1, z1]],
        "left":   [[x0, y0, z1], [x0, y0, z0], [x0, y1, z0], [x0, y1, z1]],
        "right":  [[x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0]],
        "top":    [[x0, y0, z1], [x1, y0, z1], [x1, y0, z0], [x0, y0, z0]],
        "bottom": [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
    }
    out = []
    for i, (face, pts) in enumerate(faces.items()):
        out.append(Quad(np.array(pts), texture(seed + i, 128), f"{name}_{face}"))
    return out


def build_room() -> list[Quad]:
    """Floor, ceiling and four walls, each tiled. y is down, z is depth."""
    quads: list[Quad] = []
    x, zf, zb = 3.2, -0.5, 7.5
    yt, yb = -1.7, 1.5

    quads += tile(np.array([[-x, yb, zf], [x, yb, zf], [x, yb, zb], [-x, yb, zb]]),
                  texture(1, 512, (0.9, 0.9, 1.0)), "floor", 7, 7)
    quads += tile(np.array([[-x, yt, zb], [x, yt, zb], [x, yt, zf], [-x, yt, zf]]),
                  texture(5, 512, (1.0, 1.0, 1.0)), "ceiling", 5, 5)
    quads += tile(np.array([[-x, yt, zb], [x, yt, zb], [x, yb, zb], [-x, yb, zb]]),
                  texture(2, 512, (1.0, 0.95, 0.9)), "back_wall", 6, 4)
    quads += tile(np.array([[-x, yt, zf], [-x, yt, zb], [-x, yb, zb], [-x, yb, zf]]),
                  texture(3, 512, (0.95, 1.0, 0.95)), "left_wall", 6, 4)
    quads += tile(np.array([[x, yt, zb], [x, yt, zf], [x, yb, zf], [x, yb, zb]]),
                  texture(4, 512, (1.0, 1.0, 0.92)), "right_wall", 6, 4)
    quads += tile(np.array([[x, yt, zf], [-x, yt, zf], [-x, yb, zf], [x, yb, zf]]),
                  texture(6, 512, (0.92, 0.96, 1.0)), "front_wall", 6, 4)
    return quads


def scene_a() -> list[Quad]:
    q = build_room()
    q += cuboid((-1.3, 0.95, 4.2), (0.9, 1.0, 0.9), 100, "chair")      # will move
    q += cuboid((1.6, 1.15, 3.4), (0.65, 0.65, 0.65), 200, "box")         # will be removed
    q += cuboid((0.1, 1.05, 6.0), (1.6, 0.9, 0.9), 300, "table")       # stays put
    return q


def scene_b() -> list[Quad]:
    q = build_room()
    q += cuboid((1.2, 0.95, 4.5), (0.9, 1.0, 0.9), 100, "chair")       # moved right
    q += cuboid((0.1, 1.05, 6.0), (1.6, 0.9, 0.9), 300, "table")       # unchanged
    q += cuboid((-1.7, 1.25, 3.1), (0.5, 0.5, 0.5), 400, "bag")        # new
    return q


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World to camera rotation for a camera at eye looking at target."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, forward], axis=0)


def render(quads: list[Quad], R: np.ndarray, eye: np.ndarray) -> np.ndarray:
    frame = np.full((H, W, 3), 25, dtype=np.uint8)
    K = np.array([[FOCAL, 0, W / 2.0], [0, FOCAL, H / 2.0], [0, 0, 1]])

    drawable = []
    for quad in quads:
        cam = (quad.corners - eye) @ R.T
        if np.any(cam[:, 2] < 0.35):
            continue
        projected = (cam @ K.T)
        projected = projected[:, :2] / projected[:, 2:3]
        if projected.max() > 4000 or projected.min() < -4000:
            continue
        drawable.append((float(cam[:, 2].mean()), quad, projected))

    for _depth, quad, projected in sorted(drawable, key=lambda t: -t[0]):
        th, tw = quad.tex.shape[:2]
        src = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
        try:
            Hm = cv2.getPerspectiveTransform(src, projected.astype(np.float32))
        except cv2.error:
            continue
        warped = cv2.warpPerspective(quad.tex, Hm, (W, H))
        mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), Hm, (W, H))
        frame[mask > 0] = warped[mask > 0]
    return frame


def write_video(path: Path, quads: list[Quad], jitter: float, seed: int) -> None:
    """Walk an arc across the room, looking at the middle."""
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W, H))
    target = np.array([0.0, 0.5, 4.6])
    n = 120
    for i in range(n):
        t_ = i / (n - 1)
        # A slow arc across the near half of the room, looking at the far half.
        angle = np.deg2rad(-42 + 84 * t_)
        radius = 2.2 + 0.5 * np.sin(t_ * np.pi)
        eye = np.array([
            np.sin(angle) * radius,
            -0.25 + 0.15 * np.sin(t_ * np.pi * 3),
            1.3 - np.cos(angle) * 0.9,
        ])
        eye += rng.normal(0, jitter, 3)
        R = look_at(eye, target + rng.normal(0, jitter * 0.4, 3))
        writer.write(render(quads, R, eye))
    writer.release()
    print(f"  wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=Path("videos"), type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("rendering synthetic room")
    write_video(args.out / "before.mp4", scene_a(), jitter=0.012, seed=1)
    write_video(args.out / "after.mp4", scene_b(), jitter=0.012, seed=2)

    print("\nground truth:")
    print("  chair moved from x=-1.3 to x=+1.2, about 2.5 m")
    print("  box removed from the right side")
    print("  bag added on the left")
    print("  table unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
