"""ImageNet classifiers for the paper's Fig. 4 accuracy curves (ResNet50,
EfficientNetV2, Swin, ViT). A view counts as correct when the top-1 class is
one of the ImageNet classes mapped to the object's LVIS category."""
from concurrent.futures import ThreadPoolExecutor
import math

import numpy as np
from PIL import Image


def correct(top1, category, class_map):
    """Boolean array: is each top-1 index one of ``category``'s mapped ImageNet classes?"""
    return np.isin(top1, class_map.get(category, ()))


def _load(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def crop_object(image, mask, margin=0.15):
    """Square white-padded crop around a binary object mask, with proportional margin."""
    if margin < 0:
        raise ValueError("crop margin must be nonnegative")
    bbox = mask.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    side = max(1, math.ceil(max(right - left, bottom - top) * (1 + 2 * margin)))
    x = math.floor((left + right - side) / 2)
    y = math.floor((top + bottom - side) / 2)
    out = Image.new("RGB", (side, side), "white")
    src = (max(0, x), max(0, y), min(image.width, x + side), min(image.height, y + side))
    if src[2] > src[0] and src[3] > src[1]:
        out.paste(image.crop(src), (src[0] - x, src[1] - y))
    return out


def _load_view(pair):
    image_path, mask_path, margin = pair
    image = _load(image_path)
    if margin is None:
        return image
    with Image.open(mask_path) as mask:
        return crop_object(image, mask.convert("L"), margin)


def classify_views(model_name, images, batch=128, device="cuda", log=print,
                   masks=None, crop_margin=None):
    """Top-1 ImageNet index ``[O, E, A]`` (int16) for ``images[o][e][a]`` RGB paths."""
    import timm
    import torch

    if crop_margin is not None and masks is None:
        raise ValueError("masks are required when crop_margin is set")
    model = timm.create_model(model_name, pretrained=True).to(device).eval()
    transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
    shape = (len(images), len(images[0]), len(images[0][0]))
    paths = [p for obj in images for row in obj for p in row]
    mask_paths = ([p for obj in masks for row in obj for p in row]
                  if masks is not None else [None] * len(paths))
    if len(mask_paths) != len(paths):
        raise ValueError("masks must match images")
    top1 = np.empty(len(paths), dtype=np.int16)
    with ThreadPoolExecutor(8) as pool, torch.inference_mode():
        for start in range(0, len(paths), batch):
            loaded = pool.map(_load_view, zip(paths[start:start + batch],
                                              mask_paths[start:start + batch],
                                              [crop_margin] * len(paths[start:start + batch])))
            x = torch.stack([transform(image) for image in loaded]).to(device)
            top1[start:start + len(x)] = model(x).argmax(1).cpu().numpy()
            if (start // batch) % 50 == 0:
                log(f"[classify] {model_name} {start + len(x)}/{len(paths)}")
    return top1.reshape(shape)
