"""Put a name on each change by going back to the pictures.

The geometry says a lump of space gained or lost an object. It cannot say what
the object was, because by that point everything is voxels.

So each cluster is projected back into the frames of the session that saw it,
the frame with the best view is chosen, the cluster's bounding box is cropped
out of that frame, and CLIP scores the crop against a vocabulary of indoor
objects.

Going back to the session that saw the object matters. A removed object exists
only in session A, so cropping it out of a session B frame would return a
picture of whatever is there now.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import CLIPModel, CLIPProcessor

from .changes import Cluster
from .occupancy import SessionMap


def _as_tensor(output):
    """Unwrap a CLIP feature call.

    transformers 5 returns a BaseModelOutputWithPooling from get_text_features
    and get_image_features, where earlier versions returned the tensor itself.
    Handle both so the project is not pinned to one release.
    """
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output


class Labeller:
    """Zero-shot naming of change clusters from image crops."""

    def __init__(self, cfg: dict, image_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  loading CLIP ({self.device})")
        self.model = CLIPModel.from_pretrained(cfg["model"]).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(cfg["model"])
        self.vocabulary = list(cfg["vocabulary"])
        self.min_crop = int(cfg["min_crop_px"])
        # A softmax over N labels sits at 1/N when the model has no idea. Below
        # a small multiple of that the answer is noise, and reporting it as a
        # name would be worse than admitting ignorance, because a confident
        # wrong label reads as a finding.
        self.confidence_floor = (float(cfg.get("min_confidence_ratio", 2.0))
                                 / max(len(self.vocabulary), 1))
        self.image_dir = image_dir
        self._text_features = self._encode_text()

    def _encode_text(self) -> torch.Tensor:
        prompts = [f"a photo of {v}" for v in self.vocabulary]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = _as_tensor(self.model.get_text_features(**inputs))
        return features / features.norm(dim=-1, keepdim=True)

    def _best_view(self, cluster: Cluster, session: SessionMap,
                   camera) -> tuple[str, tuple[int, int, int, int]] | None:
        """Find the frame that sees this cluster most squarely, and the crop box.

        Scored on how much of the frame the cluster fills, so a close, head-on
        view beats a glimpse at the edge.
        """
        world = (cluster.voxels.astype(np.float64) + 0.5) * self._voxel_size
        best = None
        best_score = 0.0

        for frame in session.frames:
            R_wc, centre = frame["R_wc"], frame["centre"]
            cam = (world - centre) @ R_wc  # world to camera is the transpose
            in_front = cam[:, 2] > 1e-6
            if in_front.sum() < 8:
                continue
            cam = cam[in_front]

            projected = np.array(camera.img_from_cam(cam), dtype=np.float64)
            w, h = int(camera.width), int(camera.height)
            inside = ((projected[:, 0] >= 0) & (projected[:, 0] < w) &
                      (projected[:, 1] >= 0) & (projected[:, 1] < h))
            if inside.sum() < 8:
                continue

            pts = projected[inside]
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            box_w, box_h = x1 - x0, y1 - y0
            if box_w < self.min_crop or box_h < self.min_crop:
                continue

            visible_fraction = inside.sum() / len(world)
            area_fraction = (box_w * box_h) / (w * h)
            score = visible_fraction * min(area_fraction * 6.0, 1.0)
            if score > best_score:
                best_score = score
                pad_x, pad_y = box_w * 0.12, box_h * 0.12
                best = (frame["name"], (
                    int(max(0, x0 - pad_x)), int(max(0, y0 - pad_y)),
                    int(min(w, x1 + pad_x)), int(min(h, y1 + pad_y))))

        return best

    def label_all(self, clusters: list[Cluster], map_a: SessionMap,
                  map_b: SessionMap, camera, voxel_size: float) -> None:
        """Name every cluster in place."""
        self._voxel_size = voxel_size
        for cluster in clusters:
            # Removals are only visible in the session that still had the object.
            session = map_a if cluster.kind in ("removed",) else map_b
            if cluster.kind == "moved":
                session = map_a if cluster.evidence.get("role") == "from" else map_b

            view = self._best_view(cluster, session, camera)
            if view is None:
                cluster.label = "unlabelled"
                cluster.evidence["reason"] = "no frame sees this region closely enough"
                continue

            name, box = view
            label, confidence = self._classify(name, box)
            if confidence < self.confidence_floor:
                cluster.label = "unidentified object"
                cluster.confidence = confidence
                cluster.evidence["reason"] = (
                    f"best guess '{label}' scored {confidence:.3f}, at or below "
                    f"chance for {len(self.vocabulary)} labels")
            else:
                cluster.label = label
                cluster.confidence = confidence
            cluster.evidence["frame"] = name
            cluster.evidence["crop"] = box

    def _classify(self, image_name: str, box) -> tuple[str, float]:
        path = self.image_dir / image_name
        with PILImage.open(path) as img:
            crop = img.convert("RGB").crop(box)
        inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = _as_tensor(self.model.get_image_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True)
        similarity = (features @ self._text_features.T).squeeze(0)
        probabilities = similarity.softmax(dim=-1)
        index = int(probabilities.argmax())
        return self.vocabulary[index], float(probabilities[index])
