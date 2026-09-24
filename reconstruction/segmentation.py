"""Use SAM3 to turn an RGB image and text query into SAM 3D foreground masks."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


class SAM3Preprocessor:
    """Load SAM3 once, then save its highest-confidence text-match masks."""

    def __init__(self, checkpoint="/data/checkpoints/sam3/sam3.pt", threshold=0.0, max_masks=1):
        import torch

        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing SAM3 checkpoint: {self.checkpoint}. Run scripts/download_models.sh."
            )
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if type(max_masks) is not int or max_masks < 1:
            raise ValueError("max_masks must be a positive integer")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run scripts/check_dependencies.sh.")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(checkpoint_path=str(self.checkpoint), load_from_HF=False)
        model.eval()
        self.threshold = threshold
        self.max_masks = max_masks
        self.processor = Sam3Processor(model, device="cuda", confidence_threshold=threshold)

    @staticmethod
    def highest_confidence_indices(scores, mask_count, threshold, max_masks):
        """Return up to ``max_masks`` threshold-passing candidates by score."""
        scores = np.asarray(scores).reshape(-1)
        if len(scores) != mask_count:
            raise ValueError("SAM3 returned a different number of scores and masks")
        eligible = np.flatnonzero(np.isfinite(scores) & (scores >= threshold))
        # A stable sort makes equal-confidence ties deterministic.
        ranked = eligible[np.argsort(-scores[eligible], kind="stable")]
        return ranked[:max_masks].tolist()

    @staticmethod
    def _numpy_prediction(prediction, image_size, include_spatial=True):
        """Move one SAM3 prediction to CPU without losing raw candidates."""
        width, height = image_size
        scores = prediction["scores"].detach().float().cpu().numpy().reshape(-1)
        if not include_spatial:
            return {"scores": scores}
        masks = prediction["masks"].detach().cpu().numpy()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        masks = masks.astype(bool)
        if masks.size:
            masks = masks.reshape(-1, height, width)
        else:
            masks = np.empty((0, height, width), dtype=bool)
        boxes = prediction["boxes"].detach().float().cpu().numpy()
        boxes = boxes.reshape(-1, 4) if boxes.size else np.empty((0, 4), dtype=boxes.dtype)
        return {"scores": scores, "boxes": boxes, "masks": masks}

    def detect_many(self, image, queries, *, spatial_queries=None):
        """Run several text queries while encoding ``image`` only once.

        The returned mapping preserves the input query order.  This matters for
        closed-vocabulary experiments, where every class must be scored against
        the exact same image embedding. By default every result includes masks
        and boxes. ``spatial_queries`` can restrict those large arrays to prompts
        whose localization output is actually needed.
        """
        queries = list(queries)
        if not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError("queries must be a nonempty sequence of nonempty strings")
        if len(set(queries)) != len(queries):
            raise ValueError("queries must not contain duplicates")
        spatial_queries = set(queries if spatial_queries is None else spatial_queries)
        if not spatial_queries.issubset(queries):
            raise ValueError("spatial_queries must be a subset of queries")
        import torch

        detections = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            for query in queries:
                prediction = self.processor.set_text_prompt(state=state, prompt=query)
                detections[query] = self._numpy_prediction(
                    prediction, (image.width, image.height),
                    include_spatial=query in spatial_queries)
            self.processor.reset_all_prompts(state)
        return detections

    def detect(self, image, query):
        """Return every SAM3 candidate for `query`, before any confidence filter.

        `image` is a normalized RGB PIL image. The result preserves SAM3's raw
        output so callers can apply their own thresholds or scoring::

            {"scores": np.ndarray (N,) float32,
             "boxes":  np.ndarray (N, 4) float32, xyxy pixels,
             "masks":  np.ndarray (N, H, W) bool at the source resolution}

        This is the shared model call: `masks_for_query` filters and writes on
        top of it, and experiments consume the unfiltered candidates directly.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        return self.detect_many(image, [query])[query]

    def masks_for_query(self, image, query, output_dir):
        """Save and return the top threshold-passing SAM3 instances matching `query`.

        `image` is a normalized RGB PIL image.
        Output masks are binary PNGs at the source image resolution.
        """
        detection = self.detect(image, query)
        masks = detection["masks"]
        scores = detection["scores"]
        boxes = detection["boxes"]
        selected_indices = self.highest_confidence_indices(
            scores, len(masks), self.threshold, self.max_masks)
        if selected_indices and len(boxes) != len(masks):
            raise ValueError("SAM3 returned a different number of boxes and masks")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for output_index, selected_index in enumerate(selected_indices):
            path = output_dir / f"mask-{output_index:03d}.png"
            mask = masks[selected_index]
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            paths.append(path)
        (output_dir / "predictions.json").write_text(json.dumps({
            "query": query,
            "image_size": [image.width, image.height],
            "box_format": "xyxy_pixels",
            "minimum_confidence": self.threshold,
            "maximum_masks": self.max_masks,
            "selected_candidate_indices": selected_indices,
            "boxes": boxes[selected_indices].tolist(),
            "scores": scores[selected_indices].tolist(),
            "masks": [path.name for path in paths],
        }, indent=2) + "\n")
        return paths


def depth_to_pointmap(depth, fx, fy, cx, cy, *, fill_invalid=False):
    """Convert aligned metric depth into Fast-SAM3D's PyTorch3D point-map frame.

    `depth` is HxW in meters. Intrinsics must describe the same RGB/depth pixel
    grid. Invalid depth pixels become zero-valued points unless
    ``fill_invalid`` is enabled. Fast-SAM3D's intrinsics solver requires a
    finite, nonzero Z value at every pixel, so the pipeline uses a
    median-valid-depth fill for zero/NaN background pixels.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must be an HxW metric-depth array")
    if fx <= 0 or fy <= 0:
        raise ValueError("fx and fy must be positive")
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    valid = np.isfinite(depth) & (depth > 0)
    if fill_invalid:
        if not valid.any():
            raise ValueError("depth must contain at least one finite positive pixel")
        z = np.where(valid, depth, float(np.median(depth[valid])))
    else:
        z = np.where(valid, depth, 0)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack((-x, -y, z), axis=-1).astype(np.float32)
