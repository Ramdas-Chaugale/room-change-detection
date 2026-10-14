"""Dense depth per frame, fitted to the sparse reconstruction.

COLMAP gives a few hundred accurate points per frame. That is enough to say
where the camera is, and nowhere near enough to say whether a chair is still
there. A monocular depth model fills in every pixel, but only up to an unknown
scale and offset, and in a coordinate frame of its own.

The fix is to use the sparse points as ground truth. Both come from the same
frame, so fitting a scale and shift that maps predicted depth onto the sparse
depths puts the dense map into COLMAP's coordinate frame. One least squares per
frame, and the dense depth inherits whatever scale the reconstruction has.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import pipeline


class DepthEstimator:
    """Wraps a monocular depth model and the per-frame fit to sparse points."""

    def __init__(self, cfg: dict):
        device = 0 if torch.cuda.is_available() else -1
        print(f"  loading depth model ({'gpu' if device == 0 else 'cpu'})")
        self.pipe = pipeline("depth-estimation", model=cfg["model"], device=device)
        self.min_points = int(cfg["min_sparse_points"])
        self.max_depth = float(cfg["max_depth_m"])

    def predict(self, image_path) -> np.ndarray:
        """Raw model output. Affine-invariant, not metric, not yet usable."""
        with PILImage.open(image_path) as img:
            out = self.pipe(img.convert("RGB"))
        return np.asarray(out["predicted_depth"], dtype=np.float64)

    def fit_to_sparse(self, predicted: np.ndarray, pixels: np.ndarray,
                      depths: np.ndarray, scale: float = 1.0) -> np.ndarray | None:
        """Map model output onto COLMAP's scale using the sparse points.

        Depth Anything predicts inverse depth, so the fit is done in inverse
        space where the relationship is linear: 1/z = a * pred + b. Solving
        there rather than on depth directly is what makes a plain least squares
        appropriate.

        Returns a depth map in COLMAP units, or None if there is not enough
        evidence to fit one honestly.
        """
        if len(pixels) < self.min_points:
            return None

        h, w = predicted.shape
        cols = np.clip(np.round(pixels[:, 0]).astype(int), 0, w - 1)
        rows = np.clip(np.round(pixels[:, 1]).astype(int), 0, h - 1)
        sampled = predicted[rows, cols]

        target_inv = 1.0 / np.maximum(depths, 1e-6)

        # Fit twice: least squares, then drop the worst tenth and refit, so a
        # handful of bad triangulations do not drag the whole frame.
        A = np.stack([sampled, np.ones_like(sampled)], axis=1)
        solution, *_ = np.linalg.lstsq(A, target_inv, rcond=None)
        residuals = np.abs(A @ solution - target_inv)
        keep = residuals <= np.quantile(residuals, 0.9)
        if keep.sum() >= self.min_points:
            solution, *_ = np.linalg.lstsq(A[keep], target_inv[keep], rcond=None)

        a, b = float(solution[0]), float(solution[1])
        inv_depth = a * predicted + b

        # Anything at or behind the camera plane is meaningless.
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(inv_depth > 1e-6, 1.0 / inv_depth, np.nan)

        # Into roughly metric units before any distance test, since the
        # reconstruction's own scale is arbitrary.
        depth = depth * scale
        depth[depth <= 0] = np.nan
        depth[depth > self.max_depth] = np.nan

        valid = np.isfinite(depth).mean()
        if valid < 0.2:
            return None
        return depth

    def resize_to(self, depth: np.ndarray, width: int, height: int) -> np.ndarray:
        """Depth models output at their own resolution, not the frame's."""
        if depth.shape == (height, width):
            return depth
        import cv2
        return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
