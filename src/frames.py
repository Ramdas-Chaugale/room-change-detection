"""Pull frames out of the two phone videos.

Both sessions end up in one folder with a prefix on the filename, because the
reconstruction in sfm.py runs over both at once and needs them side by side.
The prefix is how everything downstream knows which session a frame came from.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def _sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Low means blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract(video_path: Path, out_dir: Path, session: str, cfg: dict) -> list[Path]:
    """Sample frames from one video, drop the blurry ones, write them out.

    Returns the frames actually kept, in order.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / float(cfg["fps"]))))
    max_side = int(cfg["max_side"])
    blur_threshold = float(cfg["blur_threshold"])

    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[Path] = []
    dropped_blurry = 0
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step != 0:
            index += 1
            continue
        index += 1

        h, w = frame.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if _sharpness(gray) < blur_threshold:
            dropped_blurry += 1
            continue

        name = f"{session}_{len(kept):05d}.jpg"
        path = out_dir / name
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept.append(path)

    cap.release()
    print(f"  {session}: kept {len(kept)} frames, dropped {dropped_blurry} blurry")
    if len(kept) < 10:
        raise RuntimeError(
            f"only {len(kept)} usable frames from {video_path.name}. Record for longer, "
            f"move more slowly, or lower frames.blur_threshold in config.yaml"
        )
    return kept


def session_of(image_name: str) -> str:
    """Recover the session from a frame filename written by extract()."""
    return image_name.split("_", 1)[0]
