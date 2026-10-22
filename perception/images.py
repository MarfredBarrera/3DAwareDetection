"""Images on the worker pipe: base64 PNG strings."""
import base64
import io
from pathlib import Path


def to_pil(image, mode="RGB"):
    """PIL image from a path, PIL image, or numpy array (HxWx3 uint8 or HxW mask)."""
    import numpy as np
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return source.convert(mode)
    if isinstance(image, Image.Image):
        return image.convert(mode)
    array = np.asarray(image)
    if array.dtype == bool or (mode == "L" and array.ndim == 2):
        array = np.where(array > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(array.astype(np.uint8)).convert(mode)


def encode(image, mode="RGB"):
    buffer = io.BytesIO()
    to_pil(image, mode).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode(text, mode="RGB"):
    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(text))) as image:
        return image.convert(mode)
