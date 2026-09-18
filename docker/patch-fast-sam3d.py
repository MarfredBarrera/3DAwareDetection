"""Portability fixes for the pinned Fast-SAM3D revision (before installation)."""
from pathlib import Path
import sys


def patch(root):
    root = Path(root)
    dino = root / "model/backbone/dit/embedder/dino.py"
    source = dino.read_text()
    source = source.replace(
        "/data3/wmq/Fast-sam3d-objects/checkpoints/torch-cache/hub/facebookresearch_dinov2_main",
        "/opt/src/dinov2",
    )
    # The condition-embedder weights are loaded strictly from SAM3D's checkpoint.
    # Avoid downloading a second set of pretrained DINO weights during construction.
    source = source.replace("backbone_kwargs = {}", 'backbone_kwargs = {"pretrained": False}')
    dino.write_text(source)

    model = root / "model/backbone/generator/flow_matching/model.py"
    source = model.read_text()
    # Upstream imports/registers a solver it never defines. Our configuration
    # disables easy mode; leave that unsupported solver unregistered.
    source = source.replace("    Euler_easy_ss,\n", "")
    source = source.replace('        "euler_easy_ss":Euler_easy_ss,\n', "")
    model.write_text(source)


if __name__ == "__main__":
    for root in sys.argv[1:]:
        patch(root)
