"""Structure from motion over both sessions at once.

This is the decision the whole project rests on. Rather than reconstructing each
recording separately and then trying to register the two clouds together, every
frame from both videos goes into a single COLMAP problem.

The parts of the room that did not change are seen in both sessions, so COLMAP
matches across them on its own and the two sessions land in one coordinate frame
with no alignment step. The unchanged scene is the registration.

Doing it the other way round means solving a point cloud registration problem
where a chunk of the scene has genuinely moved, which is exactly the case where
registration is least reliable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pycolmap

from .frames import session_of


def reconstruct(image_dir: Path, work_dir: Path, cfg: dict,
                reuse: bool = True) -> pycolmap.Reconstruction:
    """Run feature extraction, matching and mapping over every frame.

    Returns the largest reconstruction COLMAP produces. An existing model in
    work_dir is reused unless reuse is False, because this stage dominates the
    runtime and nothing downstream of it changes the geometry.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"

    existing = sparse_dir / "0"
    if reuse and (existing / "cameras.bin").exists():
        rec = pycolmap.Reconstruction(existing)
        print(f"  reusing cached reconstruction: {rec.num_reg_images()} frames, "
              f"{rec.num_points3D()} points")
        print("  delete the colmap folder to force a rebuild")
        _check_both_sessions(rec, cfg, rec.num_reg_images())
        return rec

    if database.exists():
        database.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    camera_mode = (pycolmap.CameraMode.SINGLE if cfg.get("single_camera", True)
                   else pycolmap.CameraMode.AUTO)

    print("  extracting features")
    extraction = pycolmap.FeatureExtractionOptions()
    # Matching cost grows with the square of the feature count, and a room does
    # not need 20,000 keypoints per frame to reconstruct. Capping this is the
    # single biggest lever on runtime.
    max_features = int(cfg.get("max_features", 4096))
    try:
        extraction.sift.max_num_features = max_features
    except AttributeError:
        pass
    pycolmap.extract_features(database, image_dir, camera_mode=camera_mode,
                              extraction_options=extraction)

    matcher = cfg.get("matcher", "exhaustive")
    print(f"  matching ({matcher})")
    if matcher == "exhaustive":
        # Exhaustive is what connects session A to session B. A sequential
        # matcher only looks at temporal neighbours, so it would build two
        # reconstructions that never meet.
        pycolmap.match_exhaustive(database)
    elif matcher == "vocabtree":
        pycolmap.match_vocabtree(database)
    else:
        raise ValueError(f"unknown matcher {matcher}, use exhaustive or vocabtree")

    print("  mapping")
    reconstructions = pycolmap.incremental_mapping(database, image_dir, sparse_dir)
    if not reconstructions:
        raise RuntimeError(
            "COLMAP could not reconstruct the scene. Usually this means the two "
            "recordings do not overlap enough. Walk the same route both times and "
            "keep plenty of unchanged background in view."
        )

    rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    total = len(list(image_dir.glob("*.jpg")))
    registered = rec.num_reg_images()
    print(f"  registered {registered} of {total} frames, {rec.num_points3D()} sparse points")

    _check_both_sessions(rec, cfg, total)
    return rec


def _check_both_sessions(rec: pycolmap.Reconstruction, cfg: dict, total: int) -> None:
    """Fail loudly if the two sessions did not end up in the same model.

    Silently producing a reconstruction containing only session A would make
    every later stage report the whole of session B as new, which looks like a
    result and is not one.
    """
    counts: dict[str, int] = {}
    for image in rec.images.values():
        s = session_of(image.name)
        counts[s] = counts.get(s, 0) + 1

    print(f"  frames per session: {counts}")
    if len(counts) < 2:
        raise RuntimeError(
            "only one session made it into the reconstruction, so the two recordings "
            "were never linked. There is not enough shared, unchanged scene between "
            "them. Re-record with more overlap."
        )

    fraction = rec.num_reg_images() / max(total, 1)
    minimum = float(cfg.get("min_registered_fraction", 0.6))
    if fraction < minimum:
        print(f"  WARNING: only {fraction:.0%} of frames registered. Results will be patchy.")


def camera_rays(camera: pycolmap.Camera, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit-ish ray directions in camera coordinates for a grid of pixels.

    Returns the pixel coordinates sampled and the matching directions, with z
    normalised to 1 so a direction multiplied by a depth gives a camera-space
    point directly.
    """
    w, h = int(camera.width), int(camera.height)
    xs = np.arange(0, w, stride, dtype=np.float64)
    ys = np.arange(0, h, stride, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # cam_from_img gives normalised image coordinates, undoing the intrinsics
    # and any distortion the camera model carries.
    normalised = np.array(camera.cam_from_img(pixels), dtype=np.float64)
    directions = np.concatenate([normalised, np.ones((len(normalised), 1))], axis=1)
    return pixels, directions


def pose_of(image: pycolmap.Image) -> tuple[np.ndarray, np.ndarray]:
    """World-from-camera rotation and the camera centre in world coordinates."""
    cam_from_world = image.cam_from_world()
    R = cam_from_world.rotation.matrix()
    t = np.asarray(cam_from_world.translation, dtype=np.float64)
    R_wc = R.T
    centre = -R_wc @ t
    return R_wc, centre


def sparse_depths(rec: pycolmap.Reconstruction, image: pycolmap.Image) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates and camera-space depths of the sparse points in a frame.

    These are the anchors that give monocular depth its scale in depth.py.
    """
    pixels: list[tuple[float, float]] = []
    depths: list[float] = []
    cam_from_world = image.cam_from_world()

    for p2d in image.points2D:
        if not p2d.has_point3D():
            continue
        pid = p2d.point3D_id
        if pid not in rec.points3D:
            continue
        point3d = rec.points3D[pid]
        xyz_cam = cam_from_world * np.asarray(point3d.xyz, dtype=np.float64)
        z = float(xyz_cam[2])
        if z <= 0:
            continue
        pixels.append((float(p2d.xy[0]), float(p2d.xy[1])))
        depths.append(z)

    if not pixels:
        return np.zeros((0, 2)), np.zeros((0,))
    return np.asarray(pixels, dtype=np.float64), np.asarray(depths, dtype=np.float64)


def scene_scale(rec: pycolmap.Reconstruction, target_median_depth: float = 3.0) -> float:
    """Factor that puts the reconstruction into roughly room-sized units.

    Structure from motion is only correct up to scale, so COLMAP's units are
    arbitrary and depend on the scene. The thresholds in config.yaml are written
    in metres, so without this a 6 metre room might reconstruct at 30 units and
    every distance test would be wrong by a factor of five.

    The median camera-to-point distance is used because it survives the few
    wildly mistriangulated points every reconstruction contains.
    """
    depths: list[float] = []
    for image in rec.images.values():
        _pixels, d = sparse_depths(rec, image)
        if len(d):
            depths.append(float(np.median(d)))
    if not depths:
        return 1.0
    median = float(np.median(depths))
    if median <= 1e-9:
        return 1.0
    return target_median_depth / median
