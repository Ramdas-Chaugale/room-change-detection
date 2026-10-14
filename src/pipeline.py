"""Run the whole thing, start to finish."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import changes as changes_mod
from . import frames as frames_mod
from . import report as report_mod
from . import sfm as sfm_mod
from .depth import DepthEstimator
from .occupancy import SessionMap
from .semantics import Labeller


def run(video_a: Path, video_b: Path, out_dir: Path, cfg: dict,
        skip_semantics: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "frames"
    work_dir = out_dir / "colmap"

    print("\n[1/6] extracting frames")
    frames_mod.extract(video_a, image_dir, "a", cfg["frames"])
    frames_mod.extract(video_b, image_dir, "b", cfg["frames"])

    print("\n[2/6] joint reconstruction over both sessions")
    rec = sfm_mod.reconstruct(image_dir, work_dir, cfg["sfm"])

    print("\n[3/6] dense depth, scaled to the reconstruction")
    scale = sfm_mod.scene_scale(rec)
    print(f"  scene scale factor {scale:.4f} "
          f"(reconstruction units to approximate metres)")
    estimator = DepthEstimator(cfg["depth"])
    voxel_size = float(cfg["occupancy"]["voxel_size"])
    maps = {
        "a": SessionMap(voxel_size, cfg["occupancy"]),
        "b": SessionMap(voxel_size, cfg["occupancy"]),
    }

    camera = None
    skipped = 0
    stride = int(cfg["occupancy"]["ray_stride"])
    max_depth = float(cfg["depth"]["max_depth_m"])

    for image in tqdm(list(rec.images.values()), desc="  fusing frames"):
        session = frames_mod.session_of(image.name)
        if session not in maps:
            continue
        cam = rec.cameras[image.camera_id]
        camera = cam

        pixels, directions = sfm_mod.camera_rays(cam, stride)
        sparse_px, sparse_depth = sfm_mod.sparse_depths(rec, image)

        predicted = estimator.predict(image_dir / image.name)
        depth_map = estimator.fit_to_sparse(predicted, sparse_px, sparse_depth, scale)
        if depth_map is None:
            skipped += 1
            continue
        depth_map = estimator.resize_to(depth_map, int(cam.width), int(cam.height))

        rows = np.clip(pixels[:, 1].astype(int), 0, depth_map.shape[0] - 1)
        cols = np.clip(pixels[:, 0].astype(int), 0, depth_map.shape[1] - 1)
        depths = depth_map[rows, cols]

        R_wc, centre = sfm_mod.pose_of(image)
        # Camera centres live in the same arbitrary units as the depths did,
        # so they get the same treatment or nothing lines up.
        maps[session].add_frame(image.name, R_wc, centre * scale,
                                directions, depths, max_depth)

    if skipped:
        print(f"  skipped {skipped} frames: too few sparse points to anchor the "
              f"depth fit, or too little of the result survived the depth limit")

    for key, session_map in maps.items():
        session_map.finalise()
        print(f"  session {key}: {session_map.summary()}")

    if not maps["a"].frames or not maps["b"].frames:
        raise RuntimeError("one session produced no usable depth, cannot compare")

    print("\n[4/6] differencing occupancy against free space")
    clusters = changes_mod.detect(maps["a"], maps["b"], voxel_size, cfg["changes"])
    print(f"  {len(clusters)} change cluster(s) after filtering")

    print("\n[5/6] naming the changes")
    if clusters and not skip_semantics:
        labeller = Labeller(cfg["semantics"], image_dir)
        labeller.label_all(clusters, maps["a"], maps["b"], camera, voxel_size)
        clusters = changes_mod.match_moves(clusters, cfg["changes"])
    elif skip_semantics:
        print("  skipped (--no-semantics)")

    print("\n[6/6] writing outputs")
    scene_path = out_dir / "scene.html"
    report_path = out_dir / "report.md"
    json_path = out_dir / "changes.json"

    report_mod.write_scene(scene_path, maps["a"], maps["b"], clusters, voxel_size)
    text = report_mod.write_report(report_path, clusters, maps["a"], maps["b"], voxel_size)
    report_mod.write_json(json_path, clusters)

    print(f"\n  {scene_path}")
    print(f"  {report_path}")
    print(f"  {json_path}")
    print("\n" + "-" * 62)
    print(text)
    print("-" * 62)

    return {"clusters": clusters, "scene": scene_path, "report": report_path}
