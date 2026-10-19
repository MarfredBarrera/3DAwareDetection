"""Frozen DINOv2 and CLIP image encoders (paper Fig. 1).

Each encoder gives one embedding per image: DINOv2's layer-normed CLS token and
CLIP's projected image embedding; ViewActive concatenates them. Encoders are
frozen, so training reads features cached once per rendered view.
"""
from concurrent.futures import ThreadPoolExecutor
import re

import numpy as np
from PIL import Image


def prepare(image, mask=None):
    """RGB image with the background outside ``mask`` set to white (renders are white-background)."""
    image = image.convert("RGB")
    if mask is None:
        return image
    white = Image.new("RGB", image.size, "white")
    return Image.composite(image, white, mask.convert("L").point(lambda v: 255 if v else 0))


def _load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def encoder_id(dino, clip):
    return re.sub(r"[^A-Za-z0-9]+", "-", f"{dino}+{clip}").strip("-")


class Encoders:
    def __init__(self, dino="facebook/dinov2-base", clip="openai/clip-vit-base-patch16", device="cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection

        self.torch, self.device = torch, torch.device(device)
        self.dino_processor = AutoImageProcessor.from_pretrained(dino)
        self.dino = AutoModel.from_pretrained(dino).to(self.device).eval()
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip)
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip).to(self.device).eval()
        self.dim = self.dino.config.hidden_size + self.clip.config.projection_dim
        self.names = (dino, clip)

    def encode(self, images):
        """``[B, dim]`` float32 tensor on the encoder device for a list of PIL images."""
        with self.torch.inference_mode():
            dino = self.dino_processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
            clip = self.clip_processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
            return self.torch.cat([self.dino(pixel_values=dino).pooler_output,
                                   self.clip(pixel_values=clip).image_embeds], dim=1).float()


def cache_features(path, done_path, images, encoders, batch=128, log=print):
    """Fill the fp16 memmap ``path`` ``[O, E, A, dim]``; ``images[o][e][a]`` are RGB paths.

    ``done_path`` holds one flag per object, so an interrupted run resumes.
    """
    shape = (len(images), len(images[0]), len(images[0][0]), encoders.dim)
    mode = "r+" if path.is_file() else "w+"
    features = np.lib.format.open_memmap(path, mode=mode, dtype=np.float16, shape=shape)
    done = np.load(done_path) if done_path.is_file() else np.zeros(len(images), dtype=bool)
    todo = [o for o in range(len(images)) if not done[o]]
    step = max(1, 8 * batch // (shape[1] * shape[2]))  # objects per flush
    for start in range(0, len(todo), step):
        chunk = todo[start:start + step]
        paths = [p for o in chunk for row in images[o] for p in row]
        with ThreadPoolExecutor(8) as pool:
            loaded = list(pool.map(_load_rgb, paths))
        vectors = [encoders.encode(loaded[i:i + batch]).cpu().numpy()
                   for i in range(0, len(loaded), batch)]
        vectors = np.concatenate(vectors).reshape(len(chunk), shape[1], shape[2], -1)
        features[chunk] = vectors.astype(np.float16)
        features.flush()
        done[chunk] = True
        np.save(done_path, done)
        log(f"[features] {int(done.sum())}/{len(images)} objects")
    return features
